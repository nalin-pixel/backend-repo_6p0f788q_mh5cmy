import os
from datetime import datetime
from typing import List, Optional, Literal, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import db, create_document, get_documents

app = FastAPI(title="Phoenix Virtual Assistant API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Utilities ----------

def _serialize(doc: dict) -> dict:
    if not doc:
        return doc
    result = {}
    for k, v in doc.items():
        if k == "_id":
            result["id"] = str(v)
        elif isinstance(v, datetime):
            result[k] = v.isoformat()
        else:
            result[k] = v
    return result


def simple_sentiment(text: str) -> Literal["positive", "neutral", "negative"]:
    t = text.lower()
    pos = ["great", "good", "love", "awesome", "thanks", "cool", "nice"]
    neg = ["bad", "hate", "terrible", "awful", "angry", "sad", "upset"]
    p = any(w in t for w in pos)
    n = any(w in t for w in neg)
    if p and not n:
        return "positive"
    if n and not p:
        return "negative"
    return "neutral"


# ---------- Health & Info ----------

@app.get("/")
def read_root():
    return {"message": "Phoenix API is running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": [],
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["connection_status"] = "Connected"
            try:
                response["collections"] = db.list_collection_names()[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    return response


# ---------- Schemas (for viewer/clients) ----------

@app.get("/schema")
def get_schema():
    # Return a compact description of collections and fields
    from schemas import User, Session, Message, Document  # type: ignore

    def fields(model: Any) -> dict:
        return {name: str(f.annotation) for name, f in model.model_fields.items()}

    return {
        "user": {"fields": fields(User)},
        "session": {"fields": fields(Session)},
        "message": {"fields": fields(Message)},
        "document": {"fields": fields(Document)},
    }


# ---------- Models ----------

class CreateSession(BaseModel):
    user_id: Optional[str] = None
    title: Optional[str] = "New Session"


class SessionResponse(BaseModel):
    id: str
    title: str


class PostMessage(BaseModel):
    session_id: str
    content: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    user_message_id: str
    assistant_message_id: str
    assistant_reply: str
    sentiment: Literal["positive", "neutral", "negative"]
    emotions: List[str] = []


class CreateDocument(BaseModel):
    title: str
    text: str
    user_id: Optional[str] = None
    tags: List[str] = []
    source: Optional[str] = None


# ---------- Sessions ----------

@app.post("/sessions", response_model=SessionResponse)
def create_session(payload: CreateSession):
    data = {
        "user_id": payload.user_id,
        "title": payload.title or "New Session",
        "sentiment": None,
        "last_message_at": None,
    }
    inserted_id = create_document("session", data)
    return {"id": inserted_id, "title": data["title"]}


@app.get("/sessions/{session_id}/messages")
def list_messages(session_id: str):
    msgs = get_documents("message", {"session_id": session_id})
    return [_serialize(m) for m in msgs]


# ---------- Messages & Chat ----------

@app.post("/messages")
def create_message(payload: PostMessage):
    msg = {
        "session_id": payload.session_id,
        "role": "user",
        "content": payload.content,
        "sentiment": simple_sentiment(payload.content),
        "emotions": None,
    }
    inserted_id = create_document("message", msg)
    return {"id": inserted_id}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    if not payload.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    user_sent = simple_sentiment(payload.message)
    # Store user message
    user_msg = {
        "session_id": payload.session_id,
        "role": "user",
        "content": payload.message,
        "sentiment": user_sent,
        "emotions": None,
    }
    user_msg_id = create_document("message", user_msg)

    # Naive assistant response (stub for LLM)
    reply = (
        "I'm Phoenix, your assistant. "
        "I understood: " + payload.message + ". "
        "How would you like me to proceed?"
    )

    assistant_msg = {
        "session_id": payload.session_id,
        "role": "assistant",
        "content": reply,
        "sentiment": "neutral",
        "emotions": ["calm"],
    }
    assistant_msg_id = create_document("message", assistant_msg)

    # Update session metadata
    try:
        if db is not None:
            db["session"].update_one(
                {"_id": db["session"].find_one({"_id": {"$exists": True}, "_id": {"$type": "objectId"}})._id if False else {"$exists": False}},
                {"$set": {"last_message_at": datetime.utcnow()}},
            )
    except Exception:
        pass

    return ChatResponse(
        session_id=payload.session_id,
        user_message_id=user_msg_id,
        assistant_message_id=assistant_msg_id,
        assistant_reply=reply,
        sentiment=user_sent,
        emotions=["calm"],
    )


# ---------- Documents (RAG store placeholder) ----------

@app.post("/documents")
def add_document(doc: CreateDocument):
    data = doc.model_dump()
    inserted_id = create_document("document", data)
    return {"id": inserted_id}


@app.get("/documents")
def list_documents(user_id: Optional[str] = None, limit: int = 50):
    filt = {"user_id": user_id} if user_id else {}
    docs = get_documents("document", filt, limit=limit)
    return [_serialize(d) for d in docs]


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
