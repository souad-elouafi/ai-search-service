import pandas as pd
import json
import random

df = pd.read_csv("raw_data/styles.csv", on_bad_lines="skip")
df = df.dropna(subset=["productDisplayName", "masterCategory"])

df = df.head(10000)

products = []
for _, row in df.iterrows():
    products.append({
        "id": int(row["id"]),
        "name": str(row["productDisplayName"]),
        "description": f"{row.get('subCategory', '')} {row.get('baseColour', '')} {row.get('usage', '')}".strip(),
        "category": str(row["masterCategory"]),
        "price": random.choice([50, 90, 120, 150, 200, 300, 450])
    })

with open("app/data/products.json", "w", encoding="utf-8") as f:
    json.dump(products, f, ensure_ascii=False, indent=2)

print(f"{len(products)} produits ecrits dans app/data/products.json")
