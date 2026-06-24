"""
LLM client — provider-agnostic wrapper around OpenAI-compatible APIs.
"""
import os
import sys
from openai import OpenAI, APIError, RateLimitError, AuthenticationError, APIConnectionError
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    print("❌ ERREUR : GROQ_API_KEY n'est pas définie dans .env")
    sys.exit(1)

# Singleton client — instancié une seule fois
_client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

DEFAULT_MODEL = "openai/gpt-oss-20b"


def chat(
    prompt: str,
    system_prompt: str = "Tu es un assistant utile et concis.",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Send a prompt to the LLM and return the text response.
    Raises exceptions on errors (caller should handle them).
    """
    response = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content