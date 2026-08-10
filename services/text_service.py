import json
from openai import OpenAI
from app.config import OPENROUTER_API_KEY
from app.prompts.text_search_prompt import TEXT_SEARCH_SYSTEM_PROMPT

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="",
)

MODEL_NAME = "meta-llama/llama-3.3-70b-instruct"


def understand_query(user_query: str) -> dict:
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
    json_str = raw_content[start:end]

    return json.loads(json_str)