import pandas as pd
import shutil
import os
from pathlib import Path
import yaml

def process_meme_labels_v2():

    # ---------------------- LOAD CONFIG.YML ----------------------
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    excel_file = config["paths"]["excel_file"]
    source_directory = config["paths"]["source_directory"]
    rejected_directory = config["paths"]["rejected_directory"]

    image_extensions = config["processing"]["image_extensions"]
    required_cols = config["excel_required_columns"]
    output_file = config["output"]["cleaned_excel_filename"]

    # ---------------------- SETUP ----------------------
    Path(rejected_directory).mkdir(parents=True, exist_ok=True)

    try:
        # Read Excel
        df = pd.read_excel(excel_file)

        if not all(col in df.columns for col in required_cols):
            print("Error: Required columns missing.")
            print("Available columns:", list(df.columns))
            return

        df['Meme Number'] = df['Meme Number'].astype(str).str.strip()

        print("\n--- Stage 1: Identify image files not in Excel ---")

        # list images
        all_image_files = [
            f for f in os.listdir(source_directory)
            if any(f.lower().endswith(ext) for ext in image_extensions)
        ]

        excel_meme_numbers = set(df['Meme Number'])

        files_to_reject_stage1 = []
        rejected_stage1_count = 0

        # Stage 1 logic unchanged
        for image_file in all_image_files:

            base_name = os.path.splitext(image_file)[0]
            is_found = False

            for meme_id in excel_meme_numbers:
                if meme_id in base_name:
                    is_found = True
                    break

            if not is_found:
                source_path = os.path.join(source_directory, image_file)
                destination_path = os.path.join(rejected_directory, image_file)
                try:
                    shutil.move(source_path, destination_path)
                    files_to_reject_stage1.append(image_file)
                    rejected_stage1_count += 1
                except Exception as e:
                    print(f"Error moving unmatched file {image_file}: {e}")

        print(f"Stage 1 Complete: Moved {rejected_stage1_count} files.\n")

        # ---------------------- Stage 2 ----------------------

        print("--- Stage 2: Rows with missing metadata ---")

        missing_mask = (
            df['Topic of the Meme'].isna() |
            (df['Topic of the Meme'].astype(str).str.strip() == '') |
            df['Sentiment'].isna() |
            (df['Sentiment'].astype(str).str.strip() == '') |
            df['Intent'].isna() |
            (df['Intent'].astype(str).str.strip() == '')
        )

        rows_to_reject = df[missing_mask]

        if rows_to_reject.empty:
            print("No rows found with missing metadata.")
        else:
            print(f"Found {len(rows_to_reject)} rows with missing metadata.")

            moved_files_stage2 = 0
            error_files_stage2 = []

            for index, row in rows_to_reject.iterrows():
                meme_number = str(row['Meme Number']).strip()

                actual_filename = None
                source_path = None

                # lookup file by meme number
                for ext in image_extensions:
                    test_filename = meme_number + ext
                    test_path = os.path.join(source_directory, test_filename)

                    if os.path.exists(test_path):
                        actual_filename = test_filename
                        source_path = test_path
                        break

                if not source_path:
                    potential_path = os.path.join(source_directory, meme_number)
                    if os.path.exists(potential_path):
                        actual_filename = meme_number
                        source_path = potential_path

                if source_path and actual_filename:
                    destination_path = os.path.join(rejected_directory, actual_filename)
                    try:
                        shutil.move(source_path, destination_path)
                        moved_files_stage2 += 1
                    except Exception as e:
                        error_files_stage2.append((actual_filename, str(e)))
                        print(f"Error moving {actual_filename}: {e}")
                else:
                    print(f"Warning: File for Meme Number {meme_number} not found.")

            print(f"Stage 2 Complete: Moved {moved_files_stage2} files.\n")

        # ---------------------- Stage 3: Clean Excel ----------------------

        print("--- Stage 3: Cleaning Excel ---")

        df_cleaned = df[~missing_mask]
        df_cleaned.to_excel(output_file, index=False)

        print(f"Cleaned Excel saved as: {output_file}")
        print(f"Removed {len(rows_to_reject)} rows with missing metadata.")

    except FileNotFoundError:
        print(f"Error: Excel file '{excel_file}' not found.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    process_meme_labels_v2()
