# app.py
from flask import Flask, request, jsonify, Response, stream_with_context, g
from flask_cors import CORS
from openai import OpenAI
import os
import time
import uuid
import json
import traceback
from dotenv import load_dotenv

# ---- RAG retriever kamu ----
from rag.retriever import retrieve_docs

# ====== INIT ======
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

DEBUG_VERBOSE = os.getenv("DEBUG_VERBOSE", "1") == "1"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

client = OpenAI()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ====== SYSTEM PROMPT ======
SYSTEM_MSG = """Kamu asisten Mall Pelayanan Public Kota Bengkulu. Jawab dalam Bahasa Indonesia.

Format jawaban:
1. Mulai dengan ## [Judul]
2. Setiap poin baru pisahkan dengan baris kosong
3. Gunakan a. b. c. untuk sub-poin
4. Gunakan **bold** untuk penekanan
5. Pastikan ada jarak antar bagian

Contoh format:
## Cara Login Nextcloud

a. Sambungkan kabel LAN ke komputer

b. Buka browser dan akses https://drive.bcaf.co.id

c. Login dengan username dan password LAN
"""

# ====== HELPERS ======
def build_messages(question: str, docs: list[str]) -> list[dict]:
    ctx = "Informasi yang tersedia:\n\n"
    for d in docs:
        ctx += f"{d}\n\n"
    ctx += f"Pertanyaan: {question}"
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": ctx},
    ]

def sse_pack_json(obj) -> str:
    """Kirim payload SSE dalam JSON agar aman terhadap newline."""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

@app.before_request
def attach_request_id():
    g.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    g.start_time = time.time()

def log_event(event: str, **kwargs):
    payload = {
        "event": event,
        "request_id": getattr(g, "request_id", "n/a"),
        "path": request.path if request else "",
        "method": request.method if request else "",
        "elapsed_ms": round((time.time() - getattr(g, "start_time", time.time())) * 1000, 1),
        **kwargs,
    }
    try:
        print("[BE]", json.dumps(payload, ensure_ascii=False))
    except Exception:
        print("[BE]", payload)

# ====== ROUTES ======
@app.route("/", methods=["GET"])
def index():
    return "<h1>🤖 Chatbot RAG API</h1><p>POST /ask atau /ask-stream</p>"

@app.route("/ask", methods=["POST"])
def ask():
    try:
        payload = request.get_json(silent=True) or {}
        question = (payload.get("question") or "").strip()
        if not question:
            resp = jsonify({"error": "question wajib diisi"})
            resp.headers["X-Request-ID"] = g.request_id
            return resp, 400

        log_event("ask_received", question=question)

        docs = retrieve_docs(question)
        if DEBUG_VERBOSE:
            log_event("rag_docs", docs_preview=docs[:3], docs_count=len(docs))

        messages = build_messages(question, docs)
        if DEBUG_VERBOSE:
            log_event("openai_messages", messages=messages)

        oa = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.1,
        )

        if DEBUG_VERBOSE:
            # hati-hati ukuran; batasi agar console tidak banjir
            log_event("openai_raw_response", raw=str(oa)[:5000])

        answer = (oa.choices[0].message.content or "").strip()
        log_event("final_answer_nonstream", answer_preview=answer[:800], total_len=len(answer))

        response = jsonify({"answer": answer})
        response.headers["X-Request-ID"] = g.request_id
        return response

    except Exception as e:
        log_event("ask_error", error=str(e), traceback=traceback.format_exc())
        resp = jsonify({"error": str(e)})
        resp.headers["X-Request-ID"] = g.request_id
        return resp, 500

@app.route("/ask-stream", methods=["POST"])
def ask_stream():
    try:
        payload = request.get_json(silent=True) or {}
        question = (payload.get("question") or "").strip()
        if not question:
            def err():
                yield sse_pack_json({"type": "error", "message": "question wajib diisi"})
                yield sse_pack_json({"type": "done"})
            resp = Response(err(), mimetype="text/event-stream")
            resp.headers["X-Request-ID"] = g.request_id
            return resp

        log_event("ask_stream_received", question=question)

        docs = retrieve_docs(question)
        if DEBUG_VERBOSE:
            log_event("rag_docs", docs_preview=docs[:3], docs_count=len(docs))

        messages = build_messages(question, docs)
        if DEBUG_VERBOSE:
            log_event("openai_messages", messages=messages)

        @stream_with_context
        def generate():
            full_text = ""
            try:
                stream = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    temperature=0.1,
                    stream=True,
                )
                for chunk in stream:
                    # struktur SDK openai==1.x
                    choice = chunk.choices[0]
                    delta = getattr(choice, "delta", None)
                    if not delta:
                        continue
                    piece = getattr(delta, "content", None)
                    if piece is None:
                        continue

                    full_text += piece
                    if DEBUG_VERBOSE and piece.strip():
                        # potong agar tidak kebanyakan
                        log_event("stream_chunk", piece=piece[:200])

                    yield sse_pack_json({"type": "chunk", "content": piece})

            except Exception as e:
                log_event("stream_error", error=str(e), traceback=traceback.format_exc())
                yield sse_pack_json({"type": "error", "message": str(e)})

            finally:
                log_event("final_answer_stream", final_preview=full_text[:1000], total_len=len(full_text))
                yield sse_pack_json({"type": "done"})

        resp = Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )
        resp.headers["X-Request-ID"] = g.request_id
        return resp

    except Exception as e:
        log_event("ask_stream_outer_error", error=str(e), traceback=traceback.format_exc())
        def err():
            yield sse_pack_json({"type": "error", "message": str(e)})
            yield sse_pack_json({"type": "done"})
        resp = Response(err(), mimetype="text/event-stream")
        resp.headers["X-Request-ID"] = g.request_id
        return resp

# ====== MAIN ======
if __name__ == "__main__":
    # debug=True untuk hot-reload & log lebih jelas saat dev
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
