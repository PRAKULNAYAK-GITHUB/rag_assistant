import uuid
from datetime import datetime, timezone
from pathlib import Path

from tinydb import Query, TinyDB

from rag_core.config import settings

db = TinyDB(settings.tinydb_path)
documents = db.table("documents")
sessions = db.table("sessions")
messages = db.table("messages")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_session() -> str:
    session_id = str(uuid.uuid4())
    sessions.insert(
        {
            "id": session_id,
            "title": "New chat",
            "active": True,
            "created_at": now_iso(),
            "updated_at": now_iso(),
        }
    )
    set_active_session(session_id)
    return session_id


def set_active_session(session_id: str) -> None:
    Session = Query()
    sessions.update({"active": False}, Session.id.exists())
    sessions.update({"active": True, "updated_at": now_iso()}, Session.id == session_id)


def get_or_create_default_session() -> str:
    Session = Query()
    active = sessions.get(Session.active == True)  # noqa: E712
    if active:
        return active["id"]
    existing = sessions.all()
    if existing:
        set_active_session(existing[-1]["id"])
        return existing[-1]["id"]
    return create_session()


def list_sessions() -> list[dict]:
    return sorted(sessions.all(), key=lambda row: row["updated_at"], reverse=True)


def add_message(session_id: str, role: str, content: str, citations: list[dict] | None = None) -> None:
    messages.insert(
        {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "role": role,
            "content": content,
            "citations": citations or [],
            "created_at": now_iso(),
        }
    )
    Session = Query()
    updates = {"updated_at": now_iso()}
    if role == "user":
        current = sessions.get(Session.id == session_id)
        if current and current.get("title") == "New chat":
            updates["title"] = content[:60]
    sessions.update(updates, Session.id == session_id)


def get_messages(session_id: str) -> list[dict]:
    Message = Query()
    rows = messages.search(Message.session_id == session_id)
    return sorted(rows, key=lambda row: row["created_at"])


def upsert_document(record: dict) -> None:
    Document = Query()
    documents.upsert(record, Document.id == record["id"])


def get_document(document_id: str) -> dict | None:
    Document = Query()
    return documents.get(Document.id == document_id)


def list_documents() -> list[dict]:
    return sorted(documents.all(), key=lambda row: row["updated_at"], reverse=True)


def remove_document(document_id: str) -> None:
    Document = Query()
    documents.remove(Document.id == document_id)


def document_path(record: dict) -> Path:
    return Path(record["path"])
