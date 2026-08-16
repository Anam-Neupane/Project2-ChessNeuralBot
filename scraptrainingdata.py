import requests
import os
import zipfile
from tqdm import tqdm

URL = "https://database.nikonoel.fr/lichess_elite_2023-01.zip"
ZIP_PATH = "data/lichess_elite_2023-01.zip"
PGN_PATH = "data/lichess_elite_2023-01.pgn"

os.makedirs("data", exist_ok=True)

if not os.path.exists(PGN_PATH):
    if not os.path.exists(ZIP_PATH):
        print(f"Downloading {URL} ...")
        response = requests.get(URL, stream=True)
        total = int(response.headers.get('content-length', 0))
        with open(ZIP_PATH, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        print("Download complete.")
    else:
        print(f"ZIP already exists: {ZIP_PATH}")

    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall("data")
        print(f"Extracted: {zf.namelist()}")
else:
    print(f"PGN already exists: {PGN_PATH}")

print("Done! PGN file ready.")