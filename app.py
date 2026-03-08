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

DEBUG_VERBOSE = os.getenv("DEBUG_VERBOSE", "0") == "1"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
LOG_FILE = os.getenv("LOG_FILE", "app.log")

# ====== LOGGING: Minimal Console + File ======
root_logger = logging.getLogger()
for h in list(root_logger.handlers):
    root_logger.removeHandler(h)

# Console handler - hanya untuk error dan startup
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.WARNING)  # Hanya WARNING dan ERROR
console_handler.setFormatter(logging.Formatter(
    fmt="%(levelname)s: %(message)s"
))
root_logger.addHandler(console_handler)

# File handler - untuk log detail (opsional)
if LOG_FILE:
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_logger.addHandler(file_handler)

root_logger.setLevel(logging.INFO)

# Nonaktifkan werkzeug request logging
werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.setLevel(logging.ERROR)
werkzeug_logger.propagate = False

client = OpenAI()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ====== SYSTEM PROMPT ======
SYSTEM_MSG = """Kamu adalah asisten Mall Pelayanan Public Kota Bengkulu. Jawab dalam Bahasa Indonesia.

Aturan penting:
1. jangan pernah mengarang jawaban
2. jawab hanya berdasarkan informasi yang tersedia

Format jawaban:
1. Jika pada dokumen tidak menggunakan point list, kamu boleh menjawab dengan paragraf biasa
2. Jika pada dokumen menggunakan point list, kamu harus menjawab dengan point list juga
3. Setiap poin baru pisahkan dengan baris kosong
4. Gunakan a. b. c. untuk sub-poin
5. Gunakan **bold** untuk penekanan
6. Pastikan ada jarak antar bagian
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

def log_error(message: str, error: Exception = None):
    """Log error saja ke console dan file"""
    if error:
        logging.error(f"{message}: {str(error)}")
        if DEBUG_VERBOSE:
            logging.error(traceback.format_exc())
    else:
        logging.error(message)

def log_debug(message: str, **kwargs):
    """Log detail hanya jika DEBUG_VERBOSE aktif"""
    if DEBUG_VERBOSE:
        payload = {"message": message, **kwargs}
        logging.debug(json.dumps(payload, ensure_ascii=False))

def log_token_usage(prompt_tokens: int, completion_tokens: int, total_tokens: int, endpoint: str):
    """Log token usage ke console dan file"""
    msg = f"[{endpoint}] Tokens - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}"
    print(msg)
    logging.info(msg)

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

        log_debug("ask_received", question=question)

        docs = retrieve_docs(question)
        log_debug("rag_docs", docs_count=len(docs))

        messages = build_messages(question, docs)
        log_debug("openai_request", messages_count=len(messages))

        oa = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.1,
        )

        answer = (oa.choices[0].message.content or "").strip()
        log_debug("answer_generated", answer_len=len(answer))

        # Ekstrak token usage dari response
        usage = oa.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

        # Log token usage ke console
        log_token_usage(prompt_tokens, completion_tokens, total_tokens, "/ask")

        # Response dengan token info
        response_data = {
            "answer": answer,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            },
            "model": OPENAI_MODEL,
            "request_id": g.request_id
        }

        response = jsonify(response_data)
        response.headers["X-Request-ID"] = g.request_id
        return response

    except Exception as e:
        log_error("Error in /ask", e)
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

        log_debug("ask_stream_received", question=question)

        docs = retrieve_docs(question)
        log_debug("rag_docs", docs_count=len(docs))

        messages = build_messages(question, docs)
        log_debug("openai_request", messages_count=len(messages))

        @stream_with_context
        def generate():
            full_text = ""
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

            # Kirim info konteks yang dikirim ke OpenAI
            yield sse_pack_json({
                "type": "context",
                "docs_count": len(docs),
                "docs": docs,
                "model": OPENAI_MODEL,
                "system_prompt": SYSTEM_MSG.strip()
            })

            try:
                stream = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    temperature=0.1,
                    stream=True,
                    stream_options={"include_usage": True}  # Minta usage info di stream
                )
                
                for chunk in stream:
                    # Cek apakah ada usage info (biasanya di chunk terakhir)
                    if hasattr(chunk, 'usage') and chunk.usage:
                        prompt_tokens = chunk.usage.prompt_tokens
                        completion_tokens = chunk.usage.completion_tokens
                        total_tokens = chunk.usage.total_tokens
                    
                    choice = chunk.choices[0] if chunk.choices else None
                    if not choice:
                        continue
                    
                    delta = getattr(choice, "delta", None)
                    if not delta:
                        continue
                    
                    piece = getattr(delta, "content", None)
                    if piece is None:
                        continue

                    full_text += piece
                    yield sse_pack_json({"type": "chunk", "content": piece})

            except Exception as e:
                log_error("Stream error", e)
                yield sse_pack_json({"type": "error", "message": str(e)})

            finally:
                log_debug("stream_completed", answer_len=len(full_text))
                
                # Log token usage ke console
                if total_tokens > 0:
                    log_token_usage(prompt_tokens, completion_tokens, total_tokens, "/ask-stream")
                
                # Kirim usage info sebagai event terakhir sebelum done
                yield sse_pack_json({
                    "type": "usage",
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    },
                    "model": OPENAI_MODEL
                })
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
        log_error("Error in /ask-stream", e)
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
    
    print(f"🚀 Server running on http://{host}:{port}")
    print(f"🤖 Model: {OPENAI_MODEL}")
    print(f"📊 Token usage will be logged for each request")
    print("-" * 60)
    
    app.run(host=host, port=port, debug=False, threaded=True)