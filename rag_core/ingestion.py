import hashlib
import uuid
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from rag_core.config import settings
from rag_core.store import document_path, get_document, list_documents, now_iso, remove_document, upsert_document
from rag_core.vector_db import delete_vectors_for_document, vectorstore


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_upload(uploaded_file) -> tuple[str, Path]:
    document_id = str(uuid.uuid4())
    filename = Path(uploaded_file.name).name
    path = settings.upload_dir / f"{document_id}-{filename}"
    with path.open("wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return document_id, path


def load_documents(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    if suffix in {".txt", ".md", ".markdown"}:
        return TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()
    raise ValueError("Only PDF, TXT, and Markdown files are supported.")


def split(raw_documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(raw_documents)


def index_path(document_id: str, path: Path, filename: str) -> dict:
    raw_documents = load_documents(path)
    chunks = split(raw_documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": index,
            }
        )
    vectorstore().add_documents(chunks)
    record = {
        "id": document_id,
        "filename": filename,
        "path": str(path),
        "sha256": sha256(path),
        "chunk_count": len(chunks),
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
    upsert_document(record)
    return record


def index_upload(uploaded_file) -> dict:
    document_id, path = save_upload(uploaded_file)
    return index_path(document_id, path, Path(uploaded_file.name).name)


def index_saved_upload(path: Path) -> dict:
    stem = path.name
    if len(stem) > 37 and stem[36] == "-":
        document_id = stem[:36]
        filename = stem[37:]
    else:
        document_id = str(uuid.uuid4())
        filename = path.name
    return index_path(document_id, path, filename)


def list_unindexed_uploads() -> list[Path]:
    indexed_paths = {document_path(record).resolve() for record in list_documents()}
    uploads = []
    for path in settings.upload_dir.glob("*"):
        if path.is_file() and path.resolve() not in indexed_paths:
            uploads.append(path)
    return sorted(uploads, key=lambda item: item.stat().st_mtime, reverse=True)


def delete_document(document_id: str, remove_file: bool = True) -> None:
    record = get_document(document_id)
    delete_vectors_for_document(document_id)
    if record and remove_file:
        path = document_path(record)
        if path.exists():
            path.unlink()
    remove_document(document_id)


def reindex_document(document_id: str) -> dict | None:
    record = get_document(document_id)
    if not record:
        return None
    path = document_path(record)
    if not path.exists():
        return None
    delete_document(document_id, remove_file=False)
    return index_path(document_id, path, record["filename"])
