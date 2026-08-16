import sys
import requests
import os
import zipfile
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')

DEFAULT_MONTH = "2024-06"
URL_TEMPLATE = "https://database.nikonoel.fr/lichess_elite_{month}.zip"


def main() -> None:
    month = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MONTH
    url = URL_TEMPLATE.format(month=month)
    zip_path = os.path.join(DATA_DIR, f"lichess_elite_{month}.zip")
    pgn_path = os.path.join(DATA_DIR, f"lichess_elite_{month}.pgn")

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(pgn_path):
        print(f"PGN already exists: {pgn_path}")
        return

    if not os.path.exists(zip_path):
        print(f"Downloading {url} ...")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        with open(zip_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        print("Download complete.")
    else:
        print(f"ZIP already exists: {zip_path}")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(DATA_DIR)
        print(f"Extracted: {zf.namelist()}")

    print(f"Done! PGN file ready: {pgn_path}")


if __name__ == "__main__":
    main()
