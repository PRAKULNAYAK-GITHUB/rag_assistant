import json

import httpx

from rag_core.config import settings


def generate_answer_stream(prompt: str):
    payload = {
        "model": settings.ollama_chat_model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": "10m",
        "options": {"temperature": 0.2, "num_predict": 350},
    }
    with httpx.Client(timeout=settings.ollama_timeout_seconds) as client:
        with client.stream("POST", f"{settings.ollama_base_url}/api/generate", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                chunk = data.get("response", "")
                if chunk:
                    yield chunk
                if data.get("done"):
                    break


def generate_answer(prompt: str) -> str:
    return "".join(generate_answer_stream(prompt)).strip()
