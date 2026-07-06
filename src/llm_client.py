"""
LLM client — Anthropic Claude wrapper.

Streamlit-safe: raises exceptions instead of calling sys.exit().
"""
import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


class LLMConfigError(Exception):
    """Raised when LLM client cannot be initialized (missing API key, etc.)."""
    pass


# Claude Haiku 4.5 — fast, affordable, excellent at structured outputs.
# Switch to "claude-sonnet-4-6" for higher quality (more expensive).
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def _get_client() -> Anthropic:
    """Lazy-init the client so Streamlit can show friendly errors."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMConfigError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to .env locally, or to Streamlit secrets in deployment."
        )
    return Anthropic(api_key=api_key)


def chat(
    prompt: str,
    system_prompt: str = "Tu es un assistant utile et concis.",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Send a prompt to Claude and return the text response.
    
    Args:
        prompt: user message
        system_prompt: system instruction (top-level param in Anthropic API)
        temperature: sampling temperature (0.0 = deterministic, 1.0 = creative)
        max_tokens: max output tokens
        model: override the default model (optional)
    """
    client = _get_client()
    
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    
    # Anthropic response is a list of content blocks; first block is the text
    return response.content[0].text