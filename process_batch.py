import csv
import json
import os
import time
from bill_extractor import answer_bill_question, extract_bill_level1
import pandas as pd

TEST_BILLS_DIR = "data/test/bills"
LEVEL1_TEMPLATE = "data/test/csv/level1.csv"
LEVEL2_TEMPLATE = "data/test/csv/level2.csv"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_image_file(bills_dir, bill_id):
  for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]:
    path = os.path.join(bills_dir, f"{bill_id}{ext}")
    if os.path.exists(path):
      return path
  raise FileNotFoundError(f"Bill image not found for {bill_id} in {bills_dir}")


def main():
  print("=== Step 1: Processing Level 1 Extraction ===")
  df_l1 = pd.read_csv(LEVEL1_TEMPLATE)
  extracted_cache = {}

  out_l1_rows = []
  for idx, row in df_l1.iterrows():
    bill_id = str(row["bill_id"]).strip()
    img_path = find_image_file(TEST_BILLS_DIR, bill_id)
    print(f"[{idx+1}/10] Extracting Level 1: {bill_id}...")

    success = False
    for attempt in range(3):
      try:
        data = extract_bill_level1(img_path)
        extracted_cache[bill_id] = data
        json_str = json.dumps(data, separators=(",", ":"))
        out_l1_rows.append({"bill_id": bill_id, "json": json_str})
        success = True
        break
      except Exception as e:
        print(f"  Attempt {attempt+1} failed: {e}. Retrying...")
        time.sleep(3)

    if not success:
      print(f"FAILED on {bill_id}")
      out_l1_rows.append({"bill_id": bill_id, "json": "{}"})

    time.sleep(1)  # Respect free tier rate limits (15 RPM)

  # Write output/level1.csv
  out_l1_path = os.path.join(OUTPUT_DIR, "level1.csv")
  df_out_l1 = pd.DataFrame(out_l1_rows)
  df_out_l1.to_csv(out_l1_path, index=False, quoting=csv.QUOTE_MINIMAL)
  print(f"Level 1 complete: Saved to {out_l1_path}\n")

  print("=== Step 2: Processing Level 2 Q&A ===")
  df_l2 = pd.read_csv(LEVEL2_TEMPLATE)
  out_l2_rows = []

  total_q = len(df_l2)
  for idx, row in df_l2.iterrows():
    bill_id = str(row["bill_id"]).strip()
    q_id = str(row["question_id"]).strip()
    question = str(row["question"]).strip()
    img_path = find_image_file(TEST_BILLS_DIR, bill_id)

    print(f"[{idx+1}/{total_q}] Answering {bill_id} - {q_id}...")
    ext_data = extracted_cache.get(bill_id, {})

    ans = ""
    for attempt in range(3):
      try:
        ans = answer_bill_question(img_path, question, ext_data)
        break
      except Exception as e:
        print(f"  Attempt {attempt+1} error: {e}. Waiting 4s...")
        time.sleep(4)

    out_l2_rows.append({
        "bill_id": bill_id,
        "question_id": q_id,
        "question": question,
        "answer": ans if ans else "Information unavailable on the bill.",
    })
    time.sleep(1.5)  # Stay within rate limits

  # Write output/level2.csv
  out_l2_path = os.path.join(OUTPUT_DIR, "level2.csv")
  df_out_l2 = pd.DataFrame(out_l2_rows)
  df_out_l2.to_csv(out_l2_path, index=False, quoting=csv.QUOTE_MINIMAL)
  print(f"Level 2 complete: Saved to {out_l2_path}\n")
  print("Batch Run Complete. Check output/ directory.")


if __name__ == "__main__":
  main()