import json
from groq import Groq
from app.config import GROQ_API_KEY
from app.prompts.text_search_prompt import TEXT_SEARCH_SYSTEM_PROMPT

client = Groq(api_key=GROQ_API_KEY)

MODEL_NAME = "llama-3.3-70b-versatile"

AUDIO_UNDERSTAND_PROMPT = """Tu es un assistant qui comprend des requetes de recherche produit
pour une marketplace marocaine (vetements, chaussures, sacs, accessoires).

Tu recois DEUX transcriptions differentes du meme message vocal, faites par deux systemes
de reconnaissance vocale differents (elles peuvent chacune contenir des erreurs, surtout si
l'utilisateur parle en Darija, arabe marocain melange francais).

Compare les deux transcriptions, devine la VRAIE intention de l'utilisateur malgre les erreurs
possibles, et reponds UNIQUEMENT en JSON valide, sans aucun texte avant ou apres, au format :
{"category": "...", "brand": "...", "color": "...", "max_price": null, "search_text": "..."}
Si une info n'est pas mentionnee, mets null.
"search_text" doit etre une reformulation courte et claire en francais, utile pour une recherche semantique."""


def understand_query(user_query: str) -> dict:
    """Comprend une requete texte simple (utilise pour /api/search/text)."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": TEXT_SEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
        temperature=0.2,
    )
    raw_content = response.choices[0].message.content
    start = raw_content.find("{")
    end = raw_content.rfind("}") + 1
    return json.loads(raw_content[start:end])


def understand_audio_query(whisper_text: str, gemini_text: str) -> dict:
    """Comprend directement une requete audio a partir des 2 transcriptions (utilise pour /api/search/audio)."""
    user_message = f"""Transcription 1 (Whisper) : "{whisper_text}"
Transcription 2 (Gemini) : "{gemini_text}"

Comprends la demande et reponds en JSON."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": AUDIO_UNDERSTAND_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )
    raw_content = response.choices[0].message.content
    start = raw_content.find("{")
    end = raw_content.rfind("}") + 1
    return json.loads(raw_content[start:end])
