from google import genai
from app.config import GEMINI_API_KEY
from PIL import Image

client = genai.Client(api_key=GEMINI_API_KEY)

IMAGE_PROMPT = (
    "Decris ce produit en une phrase courte : categorie, couleur, type. "
    "Reponds en francais, sans phrase d'introduction."
)


def describe_image(image_path: str) -> str:
    img = Image.open(image_path)
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=[IMAGE_PROMPT, img],
    )
    return response.text
