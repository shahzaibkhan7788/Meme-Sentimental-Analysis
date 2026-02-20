
import os
import time
import yaml
import shutil
import pandas as pd

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

# ==============================
# 1. LOAD CONFIG
# ==============================
CONFIG_PATH = "config.yml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# ==============================
# 2. AZURE CONFIG
# ==============================
ACCOUNT_KEY = config["azure"]["key"]
REGION = config["azure"]["region"]
ENDPOINT = config["azure"]["endpoint"]

credentials = CognitiveServicesCredentials(ACCOUNT_KEY)
client = ComputerVisionClient(ENDPOINT, credentials)

# ==============================
# 3. PATHS
# ==============================
EXCEL_PATH = config["paths"]["excel_file"]
IMAGE_DIR = config["paths"]["source_directory"]
REJECTED_DIR = config["paths"]["rejected_directory"]

OUTPUT_EXCEL = os.path.join(
    os.path.dirname(EXCEL_PATH),
    config["output"]["cleaned_excel_filename"]
)

IMAGE_EXTENSIONS = tuple(config["processing"]["image_extensions"])
REQUIRED_COLUMNS = config["excel_required_columns"]

os.makedirs(REJECTED_DIR, exist_ok=True)

# ==============================
# 4. OCR FUNCTION
# ==============================
def extract_text_from_image(image_path):
    try:
        with open(image_path, "rb") as image_stream:
            raw_response = client.read_in_stream(
                image_stream,
                language="en",
                raw=True
            )

        operation_location = raw_response.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]

        while True:
            result = client.get_read_result(operation_id)
            if result.status not in ["notStarted", "running"]:
                break
            time.sleep(1)

        if result.status == OperationStatusCodes.succeeded:
            lines = []
            for page in result.analyze_result.read_results:
                for line in page.lines:
                    lines.append(line.text)

            return " ".join(lines)

        return ""

    except Exception as e:
        print(f"[OCR ERROR] {image_path}: {e}")
        return ""

# ==============================
# 5. LOAD & VALIDATE EXCEL
# ==============================
df = pd.read_excel(EXCEL_PATH)

missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in Excel: {missing_cols}")

if "Description" not in df.columns:
    df["Description"] = ""

# ==============================
# 6. PROCESS MEMES
# ==============================
for idx, row in df.iterrows():
    meme_number = row["Meme Number"]

    image_found = False
    image_path = None

    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(IMAGE_DIR, f"meme_{meme_number}{ext}")
        if os.path.exists(candidate):
            image_found = True
            image_path = candidate
            break

    if not image_found:
        print(f"[REJECTED] Image missing for meme {meme_number}")
        continue

    print(f"[PROCESSING] {os.path.basename(image_path)}")
    description = extract_text_from_image(image_path)

    if description.strip() == "":
        shutil.move(image_path, os.path.join(REJECTED_DIR, os.path.basename(image_path)))
        print(f"[MOVED TO REJECTED] {image_path}")
        continue

    df.at[idx, "Description"] = description

# ==============================
# 7. SAVE OUTPUT
# ==============================
df.to_excel(OUTPUT_EXCEL, index=False)
print("✅ Processing complete!")
print("📄 Output file:", OUTPUT_EXCEL)
print("here is the df:",df)
