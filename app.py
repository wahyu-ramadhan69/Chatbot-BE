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
import logging
import sys

# ---- RAG retriever kamu ----
from rag.retriever import retrieve_docs

# ====== INIT ======
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# meski ada DEBUG_VERBOSE, kita TETAP tidak akan kirim log ke console
DEBUG_VERBOSE = os.getenv("DEBUG_VERBOSE", "0") == "1"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# ====== LOGGING: ke file saja, tidak ke terminal ======
LOG_FILE = os.getenv("LOG_FILE", "app.log")

# Hapus semua handler root logger agar tidak ada StreamHandler ke stderr
root_logger = logging.getLogger()
for h in list(root_logger.handlers):
    root_logger.removeHandler(h)

# Buat file handler
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.DEBUG if DEBUG_VERBOSE else logging.INFO)
file_handler.setFormatter(logging.Formatter(
    fmt="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))

root_logger.addHandler(file_handler)
root_logger.setLevel(logging.DEBUG if DEBUG_VERBOSE else logging.INFO)

# Matikan semua logger bawaan werkzeug agar tidak mencetak "Running on ..." dll.
for name in ("werkzeug", "werkzeug.serving", "werkzeug._internal"):
    lg = logging.getLogger(name)
    lg.handlers = []           # pastikan tidak mewarisi handler ke console
    lg.propagate = False
    lg.disabled = True
    lg.setLevel(logging.CRITICAL)

# Matikan banner Flask CLI kalau ada
try:
    import flask.cli as flask_cli
    flask_cli.show_server_banner = lambda *args, **kwargs: None
except Exception:
    pass

client = OpenAI()
app = Flask(__name__)
# CORS tetap aktif
CORS(app, resources={r"/*": {"origins": "*"}})

# ====== SYSTEM PROMPT ======
SYSTEM_MSG = """Kamu adalah asisten Mall Pelayanan Public Kota Bengkulu. Jawab dalam Bahasa Indonesia.

Aturan penting:
1. jangan pernah mengarang jawaban
2. jawab hanya berdasarkan informasi yang tersedia

Format jawaban:
1. Setiap poin baru pisahkan dengan baris kosong
2. Gunakan a. b. c. untuk sub-poin
3. Gunakan **bold** untuk penekanan
4. Pastikan ada jarak antar bagian
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

def log_event(event: str, level: str = "info", **kwargs):
    payload = {
        "event": event,
        "request_id": getattr(g, "request_id", "n/a"),
        "path": request.path if request else "",
        "method": request.method if request else "",
        "elapsed_ms": round((time.time() - getattr(g, "start_time", time.time())) * 1000, 1),
        **kwargs,
    }
    if level == "debug":
        logging.getLogger().debug(json.dumps(payload, ensure_ascii=False))
    else:
        logging.getLogger().info(json.dumps(payload, ensure_ascii=False))

# ====== ROUTES ======
@app.route("/", methods=["GET"])
def index():
    # jangan print ke console—biarkan client yang melihat
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
            log_event("rag_docs", level="debug", docs_preview=docs[:3], docs_count=len(docs))

        messages = build_messages(question, docs)
        if DEBUG_VERBOSE:
            log_event("openai_messages", level="debug", messages=messages)

        oa = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.1,
        )

        if DEBUG_VERBOSE:
            log_event("openai_raw_response", level="debug", raw=str(oa)[:5000])

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
            log_event("rag_docs", level="debug", docs_preview=docs[:3], docs_count=len(docs))

        messages = build_messages(question, docs)
        if DEBUG_VERBOSE:
            log_event("openai_messages", level="debug", messages=messages)

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
                    choice = chunk.choices[0]
                    delta = getattr(choice, "delta", None)
                    if not delta:
                        continue
                    piece = getattr(delta, "content", None)
                    if piece is None:
                        continue

                    full_text += piece
                    if DEBUG_VERBOSE and piece.strip():
                        log_event("stream_chunk", level="debug", piece=piece[:200])

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
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 5000))
    app.run(host=host, port=port, threaded=True)
    # Penting: debug=False + use_reloader=False supaya werkzeug tidak cetak apa pun
    # app.run(host=host, port=port, debug=True, use_reloader=True, threaded=True)
