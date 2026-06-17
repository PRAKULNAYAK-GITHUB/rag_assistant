import streamlit as st
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchAny, MatchValue, VectorParams

from rag_core.config import settings


@st.cache_resource(show_spinner=False)
def embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=settings.fastembed_model)


@st.cache_resource(show_spinner=False)
def client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def ensure_qdrant_ready() -> None:
    try:
        client().collection_exists(settings.qdrant_collection)
    except Exception as exc:
        raise RuntimeError(
            f"Qdrant is not reachable at {settings.qdrant_url}. Start Qdrant before indexing."
        ) from exc


def ensure_collection() -> None:
    ensure_qdrant_ready()
    if client().collection_exists(settings.qdrant_collection):
        return
    probe = embeddings().embed_query("dimension probe")
    client().create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=len(probe), distance=Distance.COSINE),
    )


def vectorstore() -> Qdrant:
    ensure_collection()
    return Qdrant(
        client=client(),
        collection_name=settings.qdrant_collection,
        embeddings=embeddings(),
    )


def document_filter(document_ids: list[str] | None) -> Filter | None:
    if not document_ids:
        return None
    return Filter(
        must=[FieldCondition(key="metadata.document_id", match=MatchAny(any=document_ids))]
    )


def delete_vectors_for_document(document_id: str) -> None:
    ensure_qdrant_ready()
    if not client().collection_exists(settings.qdrant_collection):
        return
    client().delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[FieldCondition(key="metadata.document_id", match=MatchValue(value=document_id))]
        ),
    )
