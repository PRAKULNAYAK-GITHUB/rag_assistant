from rag_core.config import settings
from rag_core.vector_db import document_filter, vectorstore


def retrieve(question: str, document_ids: list[str] | None):
    return vectorstore().similarity_search_with_score(
        question,
        k=settings.retrieval_top_k,
        filter=document_filter(document_ids),
    )


def build_citations(hits) -> list[dict]:
    citations = []
    seen = set()
    for index, (doc, score) in enumerate(hits, start=1):
        metadata = doc.metadata
        key = (metadata.get("document_id"), metadata.get("page"), metadata.get("chunk_index"))
        if key in seen:
            continue
        seen.add(key)
        page = metadata.get("page")
        citations.append(
            {
                "source": index,
                "filename": metadata.get("filename", "unknown"),
                "page": page + 1 if isinstance(page, int) else None,
                "score": float(score),
                "snippet": doc.page_content[:320],
            }
        )
    return citations
