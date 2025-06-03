import os
import requests
from pathlib import Path
from subprocess import run
from tqdm import tqdm
from dotenv import load_dotenv

# === CONFIGURATION ===
COLLECTION_ID = "mhm-63_5"
ITEM_IDS = [
    "634_58_5025", "634_58_5050", "634_58_5075",
    "634_58_7525", "634_58_7550", "634_58_7575",
    "634_59_5000", "634_59_7500",
    "635_58_0025", "635_58_0050", "635_58_0075", "635_59_0000"
]
STAC_ROOT = "https://api.lantmateriet.se/stac-hojd/v1"

OUT_DIR = Path("data/raw/dem")
VRT_FILE = Path("data/raw/dem/mosaic.vrt")

OUT_DIR.mkdir(parents=True, exist_ok=True)
Path("data/derived").mkdir(parents=True, exist_ok=True)

# Load credentials from .env
load_dotenv()
GEOTORGET_USERNAME = os.getenv("GEOTORGET_USERNAME")
GEOTORGET_PASSWORD = os.getenv("GEOTORGET_PASSWORD")
AUTH = (GEOTORGET_USERNAME, GEOTORGET_PASSWORD)

# Manual fallback URLs for missing tiles
MANUAL_TILE_URLS = {
    "635_58_0050": "https://dl1.lantmateriet.se/hojd/data/grid1m/63_5/55/63500_5850_25.tif",
    "635_58_0075": "https://dl1.lantmateriet.se/hojd/data/grid1m/63_5/05/63450_5875_25.tif",
    "635_59_0000": "https://dl1.lantmateriet.se/hojd/data/grid1m/63_5/55/63500_5900_25.tif",
    "635_58_0075": "https://dl1.lantmateriet.se/hojd/data/grid1m/63_5/55/63500_5875_25.tif"
}

# === DOWNLOAD TILES ===
tile_hrefs = []
print("Fetching download links from STAC API...")
for item_id in ITEM_IDS:
    url = f"{STAC_ROOT}/collections/{COLLECTION_ID}/items/{item_id}"
    r = requests.get(url)
    if r.status_code == 200:
        item = r.json()
        href = item["assets"]["data"]["href"]
        tile_hrefs.append(href)
    elif item_id in MANUAL_TILE_URLS:
        print(f"⚠️  Tile {item_id} not found in API, using manual URL.")
        tile_hrefs.append(MANUAL_TILE_URLS[item_id])
    else:
        print(f"⚠️  Tile {item_id} not found.")

print(f"\n✅ Found {len(tile_hrefs)} tiles to download.\n")

for href in tqdm(tile_hrefs, desc="Downloading tiles"):
    filename = OUT_DIR / Path(href).name
    if filename.exists():
        continue
    r = requests.get(href, stream=True, auth=AUTH)
    r.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)


# === BUILD VRT ===
tif_files = list(OUT_DIR.glob("*.tif"))
if tif_files:
    print(f"\n📦 Building VRT from {len(tif_files)} tiles...")
    run(["gdalbuildvrt", str(VRT_FILE)] + [str(f) for f in tif_files], check=True)
    
    print("\n✅ DEM download and VRT creation complete!")
    print(f"VRT file: {VRT_FILE}")
else:
    print("❌ No .tif files found, skipping VRT creation.")
