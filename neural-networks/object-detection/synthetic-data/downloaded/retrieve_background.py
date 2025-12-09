import os
import requests
from pathlib import Path

ACCESS_KEY = ""
query = "desk office indoor classroom"
per_page = 30  # images per request
pages = 500      # how many pages you want

output_dir = Path("backgrounds_unsplash")
output_dir.mkdir(exist_ok=True)

for page in range(1, pages + 1):
    url = "https://api.unsplash.com/search/photos"
    params = {
        "query": query,
        "page": page,
        "per_page": per_page,
        "orientation": "landscape",
    }
    headers = {"Authorization": f"Client-ID {ACCESS_KEY}"}
    r = requests.get(url, params=params, headers=headers)
    r.raise_for_status()
    data = r.json()

    for i, result in enumerate(data["results"]):
        img_url = result["urls"]["regular"]
        img_id = result["id"]
        img_bytes = requests.get(img_url).content
        out_path = output_dir / f"{img_id}.jpg"
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        print("Saved", out_path)
