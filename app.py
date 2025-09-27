from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv
from rag.retriever import retrieve_docs

# Load API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

client = OpenAI()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

SYSTEM_MSG = (
    "Kamu asisten Mall Pelayanan Public Kota Bengkulu. Jawab ringkas dalam Bahasa Indonesia.\n"
    "Jika tidak tahu jawabannya, katakan 'Maaf, saya tidak tahu.'"
    "Jangan mengarang jawaban."
    "Jawaban harus berdasarkan konteks yang diberikan."
)

def build_messages(question: str, docs: list[str]) -> list[dict]:
    ctx = "Gunakan informasi berikut untuk menjawab:\n\n"
    for d in docs:
        ctx += f"{d}\n\n"
    ctx += f"Pertanyaan: {question}\nJawaban:"
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": ctx},
    ]

def sse_pack(data: str) -> str:
    return f"data: {data}\n\n"

@app.route("/", methods=["GET"])
def index():
    return "<h1>🤖 Chatbot RAG API</h1><p>POST /ask atau /ask-stream</p>"

@app.route("/ask", methods=["POST"])
def ask():
    """Endpoint tanpa streaming"""
    try:
        payload = request.get_json(silent=True) or {}
        question = (payload.get("question") or "").strip()
        if not question:
            return jsonify({"error": "question wajib diisi"}), 400

        docs = retrieve_docs(question)
        messages = build_messages(question, docs)

        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2,
        )
        answer = (resp.choices[0].message.content or "").strip()
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask-stream", methods=["POST"])
def ask_stream():
    """Endpoint streaming (SSE)"""
    try:
        payload = request.get_json(silent=True) or {}
        question = (payload.get("question") or "").strip()
        if not question:
            def err():
                yield sse_pack("[ERROR] question wajib diisi")
                yield sse_pack("[DONE]")
            return Response(err(), mimetype="text/event-stream")

        docs = retrieve_docs(question)
        messages = build_messages(question, docs)

        @stream_with_context
        def generate():
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.2,
                stream=True,
            )
            try:
                for chunk in stream:
                    choice = chunk.choices[0]
                    delta = getattr(choice, "delta", None)
                    if not delta:
                        continue
                    piece = getattr(delta, "content", None)
                    if piece is not None:
                        yield sse_pack(piece)
            except Exception as e:
                yield sse_pack(f"[ERROR] {e}")
            finally:
                yield sse_pack("[DONE]")

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    except Exception as e:
        def err():
            yield sse_pack(f"[ERROR] {e}")
            yield sse_pack("[DONE]")
        return Response(err(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)