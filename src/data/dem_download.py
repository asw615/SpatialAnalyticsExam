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
DEM_5M = Path("data/derived/dem_5m.tif")
SLOPE = Path("data/derived/slope.tif")
ASPECT = Path("data/derived/aspect.tif")

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


# === BUILD VRT & PROCESS ===
tif_files = list(OUT_DIR.glob("*.tif"))
if tif_files:
    print(f"\n📦 Building VRT from {len(tif_files)} tiles...")
    run(["gdalbuildvrt", str(VRT_FILE)] + [str(f) for f in tif_files], check=True)

    print("🌍 Warping to 5m DEM grid...")
    run([
        "gdalwarp", "-overwrite", "-tr", "5", "5", "-tap", "-r", "average",
        "-t_srs", "EPSG:3006", str(VRT_FILE), str(DEM_5M)
    ], check=True)

    print("📈 Generating slope and aspect...")
    run(["gdaldem", "slope", str(DEM_5M), str(SLOPE)], check=True)
    run(["gdaldem", "aspect", str(DEM_5M), str(ASPECT)], check=True)

    print("\n✅ All done!")
    print(f"DEM:      {DEM_5M}")
    print(f"Slope:    {SLOPE}")
    print(f"Aspect:   {ASPECT}")
else:
    print("❌ No .tif files found, skipping processing.")
