import csv
import json
import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
from PIL import Image

load_dotenv()
API_KEY = (
    os.getenv("API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or os.getenv("MODEL_API_KEY")
)
if not API_KEY:
  raise ValueError("API_KEY not found in .env file!")

genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-3.5-flash-lite"

TEST_BILLS_DIR = os.path.join("data", "test", "bills")
LEVEL1_TEMPLATE = os.path.join("data", "test", "csv", "level1.csv")
LEVEL2_TEMPLATE = os.path.join("data", "test", "csv", "level2.csv")
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LEVEL1_PROMPT = """
You are a precise utility bill extraction engine for Pakistani electricity bills (KE, LESCO, IESCO).
Extract all non-identifying billing information strictly following this JSON schema:

{
  "provider": "KE" | "LESCO" | "IESCO",
  "tariff": "string",
  "sanctioned_load_kw": number,
  "bill_month": "YYYY-MM",
  "reading_date": "YYYY-MM-DD",
  "issue_date": "YYYY-MM-DD",
  "due_date": "YYYY-MM-DD",
  "previous_reading": number,
  "current_reading": number,
  "units_consumed": number,
  "charges": [
    {"type": "energy" | "fixed" | "fpa" | "quarterly_adjustment" | "surcharge" | "meter_rent" | "subsidy" | "other", "amount": number}
  ],
  "total_charges": number,
  "taxes": [
    {"type": "gst" | "electricity_duty" | "income_tax" | "municipal_tax" | "other_tax", "amount": number}
  ],
  "total_taxes": number,
  "current_bill": number,
  "arrears": number,
  "payable_within_due_date": number,
  "payable_after_due_date": number
}

STRICT EXTRACTION RULES:
1. provider: Must be exactly "KE", "LESCO", or "IESCO". For K-Electric, map to "KE".
2. Extract numeric values as raw numbers. Remove "Rs." and commas.
3. Subsidies, relief lines, and credits (CR) MUST be negative numbers.
4. Dates must follow YYYY-MM-DD and bill_month must be YYYY-MM.
5. If payable_after_due_date has multiple tiers, return the highest amount.
6. If a field is blank, missing, or obscured by a grey box, return null (or [] for lists).
7. Do NOT include customer names, addresses, CNICs, or meter numbers.
8. Output ONLY the valid JSON object.
"""


def find_bill_image(bills_dir, bill_id):
  for ext in [
      ".jpg",
      ".png",
      ".jpeg",
      ".JPG",
      ".PNG",
      ".JPEG",
      ".webp",
      ".jfif",
  ]:
    path = os.path.join(bills_dir, f"{bill_id}{ext}")
    if os.path.exists(path):
      return path
  # Check if bill_id already includes extension
  direct_path = os.path.join(bills_dir, bill_id)
  if os.path.exists(direct_path):
    return direct_path
  raise FileNotFoundError(f"Image for bill {bill_id} not found in {bills_dir}")


def run_pipeline():
  print("--- STEP 1: PROCESSING LEVEL 1 EXTRACTION ---")
  df_l1 = pd.read_csv(LEVEL1_TEMPLATE)
  extracted_bills = {}
  l1_rows = []

  model_l1 = genai.GenerativeModel(
      MODEL_NAME, generation_config={"response_mime_type": "application/json"}
  )

  for idx, row in df_l1.iterrows():
    bill_id = str(row["bill_id"]).strip()
    img_path = find_bill_image(TEST_BILLS_DIR, bill_id)
    print(f"[{idx+1}/10] Extracting Level 1: {bill_id}...")

    img = Image.open(img_path)
    success = False
    for attempt in range(3):
      try:
        response = model_l1.generate_content([LEVEL1_PROMPT, img])
        parsed_json = json.loads(response.text)
        extracted_bills[bill_id] = parsed_json
        compact_json = json.dumps(parsed_json, separators=(",", ":"))
        l1_rows.append({"bill_id": bill_id, "json": compact_json})
        success = True
        break
      except Exception as e:
        print(f"  Attempt {attempt+1} failed ({e}). Retrying in 3s...")
        time.sleep(3)

    if not success:
      print(f"  ERROR on {bill_id}, writing fallback JSON.")
      l1_rows.append({"bill_id": bill_id, "json": "{}"})

    time.sleep(1)

  # Write output/level1.csv
  out_l1_path = os.path.join(OUTPUT_DIR, "level1.csv")
  df_out_l1 = pd.DataFrame(l1_rows)
  df_out_l1.to_csv(
      out_l1_path, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8"
  )
  print(f"✓ Saved {out_l1_path}\n")

  print("--- STEP 2: PROCESSING LEVEL 2 GROUNDED Q&A ---")
  df_l2 = pd.read_csv(LEVEL2_TEMPLATE)
  answers_map = {}  # (bill_id, question_id) -> answer

  model_l2 = genai.GenerativeModel(
      MODEL_NAME, generation_config={"response_mime_type": "application/json"}
  )

  unique_bills = df_l2["bill_id"].unique()
  for b_idx, bill_id in enumerate(unique_bills):
    bill_id = str(bill_id).strip()
    bill_questions = df_l2[df_l2["bill_id"] == bill_id]
    img_path = find_bill_image(TEST_BILLS_DIR, bill_id)
    bill_json = extracted_bills.get(bill_id, {})

    q_list_text = "\n".join([
        f"- {r['question_id']}: {r['question']}"
        for _, r in bill_questions.iterrows()
    ])

    l2_prompt = f"""
        You are a customer service assistant answering questions about this electricity bill.
        Extracted Bill Data:
        {json.dumps(bill_json)}

        Answer each of the following questions in plain English, strictly using only facts on this bill:
        {q_list_text}

        RULES:
        1. Ground each answer strictly in the bill. If a value is missing or obscured by a grey box, state that the information is unavailable on the bill. Do not guess.
        2. Show exact calculations (sums, differences, percentages, historical month counts).
        3. If a value is an estimate, label it as an estimate and state your formula.
        4. Exclude personal identifiers (no names, addresses, CNICs, or meter numbers).
        5. Output a JSON object mapping each question_id to its plain English answer string:
        {{
           "question_id": "answer text",
           ...
        }}
        """

    print(
        f"[{b_idx+1}/10] Answering all questions for bill {bill_id} via Gemini..."
    )
    img = Image.open(img_path)
    for attempt in range(3):
      try:
        resp = model_l2.generate_content([l2_prompt, img])
        q_answers = json.loads(resp.text)
        for q_id, ans in q_answers.items():
          # Clean newline characters so CSV rows stay unified
          clean_ans = str(ans).replace("\n", " ").strip()
          answers_map[(bill_id, str(q_id))] = clean_ans
        break
      except Exception as e:
        print(f"  Attempt {attempt+1} failed ({e}). Retrying in 4s...")
        time.sleep(4)

    time.sleep(1)

  # Assemble output/level2.csv in exact original order
  out_l2_rows = []
  for _, row in df_l2.iterrows():
    b_id = str(row["bill_id"]).strip()
    q_id = str(row["question_id"]).strip()
    q_txt = str(row["question"]).strip()
    ans_txt = answers_map.get(
        (b_id, q_id), "Information unavailable on the bill."
    )

    out_l2_rows.append({
        "bill_id": b_id,
        "question_id": q_id,
        "question": q_txt,
        "answer": ans_txt,
    })

  out_l2_path = os.path.join(OUTPUT_DIR, "level2.csv")
  df_out_l2 = pd.DataFrame(out_l2_rows)
  df_out_l2.to_csv(
      out_l2_path, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8"
  )
  print(f"✓ Saved {out_l2_path}\n")

  # --- AUTOMATED INTEGRITY CHECKS ---
  print("--- STEP 3: VERIFYING OUTPUT CSV INTEGRITY ---")
  check_l1 = pd.read_csv(out_l1_path)
  check_l2 = pd.read_csv(out_l2_path)

  assert len(check_l1) == 10, f"Error: Level 1 has {len(check_l1)} rows (need 10)"[cite: 3]
  assert (
      check_l1["json"].isna().sum() == 0
  ), "Error: Blank cells found in Level 1 json column!"
  assert (
      len(check_l2) == 120
  ), f"Error: Level 2 has {len(check_l2)} rows (need 120)"[cite: 3]
  assert (
      check_l2["answer"].isna().sum() == 0
  ), "Error: Blank cells found in Level 2 answer column!"
  assert (
      (check_l2["answer"].str.strip() == "").sum() == 0
  ), "Error: Blank strings found in Level 2 answers!"

  print("🎉 SUCCESS: ALL CHECKS PASSED!")
  print("output/level1.csv (10 rows) and output/level2.csv (120 rows) are ready.")


if __name__ == "__main__":
  run_pipeline()