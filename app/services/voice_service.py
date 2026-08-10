from groq import Groq
from google import genai
from app.config import GROQ_API_KEY, GEMINI_API_KEY

groq_client = Groq(api_key=GROQ_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

DOMAIN_PROMPT = (
    "Recherche de produits sur une marketplace marocaine. "
    "Vetements, chaussures, sacs, bghit, khouya, dial, li kayn, "
    "prix, dirham, taille, couleur, kayen, bghina."
)

GEMINI_AUDIO_PROMPT = (
    "Cet audio est une recherche produit sur une marketplace marocaine. "
    "L'utilisateur parle en Darija, en arabe, ou en francais (parfois melange). "
    "Ecoute et ecris en francais ce que la personne recherche, sous forme de phrase claire. "
    "Reponds uniquement avec la phrase, rien d'autre."
)


def transcribe_dual(file_path: str) -> dict:
    """Transcrit le meme audio avec Whisper ET Gemini, retourne les 2 versions brutes."""
    with open(file_path, "rb") as audio_file:
        whisper_result = groq_client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            prompt=DOMAIN_PROMPT,
        )

    uploaded = gemini_client.files.upload(file=file_path)
    gemini_response = gemini_client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=[GEMINI_AUDIO_PROMPT, uploaded],
    )

    return {
        "whisper_text": whisper_result.text,
        "gemini_text": gemini_response.text.strip(),
    }
