def build_grounded_prompt(question: str, hits, history: list[dict]) -> str:
    context_blocks = []
    for index, (doc, score) in enumerate(hits, start=1):
        filename = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page")
        page_label = f", page {page + 1}" if isinstance(page, int) else ""
        context_blocks.append(
            f"[Source {index}: {filename}{page_label}, score {score:.3f}]\n{doc.page_content}"
        )

    recent_history = "\n".join(
        f"{item['role'].title()}: {item['content']}" for item in history[-8:]
    )
    context = "\n\n".join(context_blocks) or "No retrieved context."

    return f"""You are a local retrieval-augmented assistant.
Answer only from the retrieved document context when the question depends on uploaded documents.
If the context is insufficient, say what is missing instead of guessing.
Cite sources inline using [Source N].

Recent chat:
{recent_history or "No previous messages."}

Retrieved context:
{context}

Question:
{question}

Answer:"""
