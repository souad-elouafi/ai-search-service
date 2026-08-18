from groq import Groq
from google import genai
from app.config import GROQ_API_KEY, GEMINI_API_KEY
from PIL import Image
from app.config import GROQ_MODEL_NAME

client = Groq(api_key=GROQ_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

IMAGE_ANALYSIS_PROMPT = (
    "Decris cette photo de produit en detail pour aider a rediger une annonce de vente : "
    "couleur precise, matiere/tissu apparent, style, etat visible (neuf/use/usure), "
    "details visuels notables (motifs, logo, coupe, forme). "
    "Reponds en francais, en quelques phrases factuelles, sans introduction."
)

DESCRIPTION_SYSTEM_PROMPT_FR = """Tu es un assistant qui aide les vendeurs d'une marketplace marocaine
(ChedMed) a rediger des descriptions de produits attractives et professionnelles.

Tu recois : le nom du produit, sa categorie, des mots-cles fournis par le vendeur,
et une analyse visuelle de la photo du produit.
Combine toutes ces informations pour rediger une description commerciale claire et
engageante en francais (3-5 phrases).

Regles :
- Ne jamais inventer de caracteristiques non mentionnees dans les infos fournies
- Priorise les details visuels de la photo (couleur, matiere, etat) s'ils sont disponibles
- Reste factuel mais valorise le produit
- Pas de formules genre "N'hesitez pas a nous contacter", va droit au but
- Ne mets pas de prix dans la description"""

DESCRIPTION_SYSTEM_PROMPT_DARIJA_ARABIC = """Tu es un assistant qui aide les vendeurs d'une marketplace
marocaine (ChedMed) a rediger des descriptions de produits en Darija (dialecte marocain),
ecrite en alphabet ARABE.

Tu recois : le nom du produit, sa categorie, des mots-cles fournis par le vendeur,
et une analyse visuelle de la photo du produit.
Combine toutes ces informations pour rediger une description naturelle en Darija
ecrite en caracteres arabes (3-5 phrases), comme un vendeur marocain s'adresserait a un client.

Regles :
- Ne jamais inventer de caracteristiques non mentionnees
- Priorise les details visuels de la photo (couleur, matiere, etat) s'ils sont disponibles
- Ton naturel, pas trop formel
- Ne mets pas de prix dans la description
- Reponds UNIQUEMENT en Darija ecrite en alphabet arabe"""

DESCRIPTION_SYSTEM_PROMPT_DARIJA_LATIN = """Tu es un assistant qui aide les vendeurs d'une marketplace
marocaine (ChedMed) a rediger des descriptions de produits en Darija/Arabizi (alphabet latin
avec chiffres pour les sons arabes, ex: 3ndi, 7lo, bghit).

Tu recois : le nom du produit, sa categorie, des mots-cles fournis par le vendeur,
et une analyse visuelle de la photo du produit.
Combine toutes ces informations pour rediger une description naturelle en Darija/Arabizi
(3-5 phrases), comme un vendeur marocain ecrirait sur WhatsApp.

Regles :
- Ne jamais inventer de caracteristiques non mentionnees
- Priorise les details visuels de la photo (couleur, matiere, etat) s'ils sont disponibles
- Ton naturel et decontracte
- Ne mets pas de prix dans la description
- Reponds UNIQUEMENT en Darija/Arabizi"""


def analyze_product_image(image_path: str) -> str:
    """Analyse une photo de produit et retourne une description visuelle detaillee."""
    img = Image.open(image_path)
    response = gemini_client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=[IMAGE_ANALYSIS_PROMPT, img],
    )
    return response.text.strip()


def _generate_one(system_prompt: str, product_name: str, category: str, keywords: list, image_analysis: str) -> str:
    keywords_text = ", ".join(keywords) if keywords else "aucun mot-cle fourni"
    image_text = image_analysis if image_analysis else "aucune photo fournie"

    user_message = f"""Nom du produit : {product_name}
Categorie : {category}
Mots-cles fournis par le vendeur : {keywords_text}
Analyse visuelle de la photo : {image_text}

Redige la description."""

    response = client.chat.completions.create(
        MODEL_NAME = GROQ_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.6,
    )
    return response.choices[0].message.content.strip()


def suggest_description(
    product_name: str,
    category: str,
    keywords: list = None,
    language: str = "fr",
    image_path: str = None,
) -> dict:
    """
    Genere une description produit a partir du nom, de la categorie, des mots-cles,
    et optionnellement d'une photo du produit.
    language: "fr", "darija_ar", "darija_latin", ou "all"
    """
    keywords = keywords or []

    image_analysis = ""
    if image_path:
        image_analysis = analyze_product_image(image_path)

    prompts = {
        "fr": DESCRIPTION_SYSTEM_PROMPT_FR,
        "darija_ar": DESCRIPTION_SYSTEM_PROMPT_DARIJA_ARABIC,
        "darija_latin": DESCRIPTION_SYSTEM_PROMPT_DARIJA_LATIN,
    }

    result = {}
    if language == "all":
        for lang_key, prompt in prompts.items():
            result[lang_key] = _generate_one(prompt, product_name, category, keywords, image_analysis)
    else:
        prompt = prompts.get(language, DESCRIPTION_SYSTEM_PROMPT_FR)
        result[language] = _generate_one(prompt, product_name, category, keywords, image_analysis)

    if image_analysis:
        result["image_analysis"] = image_analysis

    return result