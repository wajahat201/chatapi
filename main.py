import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pinecone client
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("medicalbot")

# embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# LLM client
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_API_KEY"]
)
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SYSTEM_PROMPT = "You are a helpful medical assistant."

# session memory
sessions = {}
def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return sessions[session_id]

def chat(query: str, session_id: str):
    messages = get_session(session_id)

    if len(messages) == 1:  # first query → add context
        query_vector = embedder.encode(query).tolist()
        results = index.query(vector=query_vector, top_k=4, include_metadata=True)
        context = "\n\n".join(m["metadata"].get("text", "") for m in results["matches"])

        query = f"Context:\n{context}\n\nQuestion:\n{query}"

    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(model=MODEL, messages=messages)
    reply = response.choices[0].message.content

    messages.append({"role": "assistant", "content": reply})
    return reply

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    chat_input = data.get("chatInput")
    session_id = data.get("sessionId")
    if not chat_input or not session_id:
        return {"reply": "Invalid input"}
    return {"reply": chat(chat_input, session_id)}
if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
