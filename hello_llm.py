"""
Day 1 — First contact with a LLM API (via Groq).
Goal: understand how to send a prompt and receive a response.

NOTE: We use Groq instead of OpenAI for free, fast inference.
The OpenAI Python SDK is compatible — we just change the base_url.
This is called "API abstraction" — and it's a powerful pattern.

Author: Vanella Lidelle Dzikang
Project: CVMatch AI
"""
import os
import sys
from openai import OpenAI, APIError, RateLimitError, AuthenticationError, APIConnectionError
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────

load_dotenv()

# On lit la clé GROQ au lieu de OPENAI
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    print("❌ ERREUR : GROQ_API_KEY n'est pas définie dans .env")
    print("   Récupère ta clé sur https://console.groq.com/keys")
    sys.exit(1)

# ⭐ LA MAGIE : on utilise le SDK OpenAI mais on pointe vers Groq
client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1"  # ← endpoint Groq
)

# Modèle Groq qu'on utilise.
# Alternatives possibles : "openai/gpt-oss-120b" (plus puissant, plus lent)
#                          "llama-3.3-70b-versatile" (encore actif, bientôt déprécié)
MODEL = "openai/gpt-oss-20b"


# ─────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────

def ask_llm(prompt: str, temperature: float = 0.7, system_prompt: str = None) -> str:
    """
    Envoie un prompt au LLM et retourne la réponse.
    
    Args:
        prompt: La question/instruction à envoyer
        temperature: Créativité (0 = déterministe, 1+ = créatif)
        system_prompt: Instruction système optionnelle
    
    Returns:
        La réponse textuelle du modèle, ou un message d'erreur explicite
    """
    if system_prompt is None:
        system_prompt = "Tu es un assistant utile et concis."
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=500
        )
        return response.choices[0].message.content
    
    except AuthenticationError:
        return "❌ Clé API invalide. Vérifie GROQ_API_KEY dans .env"
    
    except RateLimitError:
        return "⏱️  Limite de requêtes atteinte (30 req/min en free tier). Attends ~1 min."
    
    except APIConnectionError:
        return "🌐 Problème de connexion. Vérifie ta connexion internet."
    
    except APIError as e:
        return f"⚠️  Erreur API : {str(e)}"
    
    except Exception as e:
        return f"❌ Erreur inattendue : {type(e).__name__} - {str(e)}"


def ask_llm_with_metrics(prompt: str, temperature: float = 0.7) -> dict:
    """
    Comme ask_llm, mais retourne aussi les métriques (tokens consommés).
    Sur Groq free tier : le coût est de 0$ mais on track quand même les tokens
    (c'est une habitude de production : on saura migrer vers du payant facilement).
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Tu es un assistant utile et concis."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=500
        )
        
        return {
            "response": response.choices[0].message.content,
            "model": MODEL,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "tier": "Groq Free Tier — $0.00",
        }
    
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)}"}


# ─────────────────────────────────────────────────────────────
# MAIN — Tests
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(f"🚀 Test 1: Simple call (model: {MODEL})")
    print("=" * 60)
    answer = ask_llm("Explique-moi ce qu'est un LLM en 3 phrases.")
    print(answer)

    print("\n" + "=" * 60)
    print("🚀 Test 2: Call with metrics")
    print("=" * 60)
    result = ask_llm_with_metrics(
        "Liste 5 compétences clés d'un AI Engineer en 2026."
    )
    
    if "error" in result:
        print(f"❌ {result['error']}")
    else:
        print(f"Response:\n{result['response']}\n")
        print(f"🤖 Model: {result['model']}")
        print(f"📊 Tokens: {result['input_tokens']} in + {result['output_tokens']} out = {result['total_tokens']} total")
        print(f"💰 {result['tier']}")

    print("\n" + "=" * 60)
    print("🚀 Test 3: Custom system prompt")
    print("=" * 60)
    answer = ask_llm(
        prompt="Donne-moi un conseil pour devenir AI Engineer.",
        system_prompt="Tu es un mentor tech direct et exigeant qui répond en français.",
        temperature=0.8
    )
    print(answer)