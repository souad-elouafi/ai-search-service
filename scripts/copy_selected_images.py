import json
import shutil
import os

IMAGES_SOURCE = "raw_data/images"
IMAGES_DEST = "app/data/images"

os.makedirs(IMAGES_DEST, exist_ok=True)

with open("app/data/products.json", "r", encoding="utf-8") as f:
    products = json.load(f)

copied = 0
missing = 0
for p in products:
    filename = f"{p['id']}.jpg"
    src = os.path.join(IMAGES_SOURCE, filename)
    dst = os.path.join(IMAGES_DEST, filename)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        copied += 1
    else:
        missing += 1

print(f"{copied} images copiees vers {IMAGES_DEST}")
print(f"{missing} images introuvables (produits sans photo)")
