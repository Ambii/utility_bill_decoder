import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

load_dotenv()
API_KEY = os.getenv("API_KEY")
genai.configure(api_key=API_KEY)

LEVEL1_PROMPT = """
You are a precise utility bill data extraction engine for Pakistani electricity bills (K-Electric, LESCO, IESCO).
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
    {
      "type": "energy" | "fixed" | "fpa" | "quarterly_adjustment" | "surcharge" | "meter_rent" | "subsidy" | "other",
      "amount": number
    }
  ],
  "total_charges": number,
  "taxes": [
    {
      "type": "gst" | "electricity_duty" | "income_tax" | "municipal_tax" | "other_tax",
      "amount": number
    }
  ],
  "total_taxes": number,
  "current_bill": number,
  "arrears": number,
  "payable_within_due_date": number,
  "payable_after_due_date": number
}

STRICT EXTRACTION RULES:
1. provider: Must be exactly "KE", "LESCO", or "IESCO". For K-Electric / KESC, use "KE".
2. Extract numeric values as raw numbers (float or int). Remove "Rs.", commas, and currency labels.
3. Subsidies, relief lines, and credits (e.g. CR) MUST be negative numbers.
4. Dates must follow YYYY-MM-DD (e.g., 2026-04-21) and bill_month must be YYYY-MM (e.g., 2026-04).
5. Charges type: "energy", "fixed", "fpa" (for FPA/FCA), "quarterly_adjustment" (QTA), "surcharge" (PHL/FC), "meter_rent", "subsidy", "other".
6. Taxes type: "gst", "electricity_duty" (ED), "income_tax", "municipal_tax" (MUCT/KMC), "other_tax" (TV fee, extra tax).
7. If a total is not printed, return null. If taxes are combined with no breakdown, taxes = [] and set total_taxes.
8. If payable_after_due_date has multiple late payment amounts, select the highest.
9. Arrears: Previous dues carried forward. Keep printed sign; return null if blank.
10. If a field is blank, covered with grey box, or missing, set its value to null.
11. DO NOT include customer names, addresses, CNIC, account numbers, or meter numbers.
12. Output ONLY the valid JSON object. No markdown fences, no conversational text.
"""


def extract_bill_level1(image_path: str) -> dict:
  model = genai.GenerativeModel(
      model_name="gemini-3.5-flash-lite",
      generation_config={"response_mime_type": "application/json"},
  )
  img = Image.open(image_path)
  response = model.generate_content([LEVEL1_PROMPT, img])
  data = json.loads(response.text)
  return data


LEVEL2_PROMPT_TEMPLATE = """
You are an expert customer service assistant answering questions about an electricity bill.
Analyze the attached bill image and the extracted bill data:
Extracted Data:
{extracted_json}

Customer Question: "{question}"

RULES FOR ANSWERING:
1. Ground answers strictly on the bill. Do not assume outside rates or tariffs.
2. If the required value is missing, not printed, or covered by a grey box, explicitly state that the information is unavailable on the bill. Do not guess.
3. For calculations (sums, differences, percentages, historical month counts above thresholds), show the exact numbers from the bill.
4. If a value is an estimate, explicitly label it as an estimate and explain how it was calculated.
5. Do NOT include personal identifiers (no names, addresses, CNIC, account/consumer numbers).
6. Answer concisely and professionally in plain English.
7. Return only the final text answer.
"""


def answer_bill_question(
    image_path: str, question: str, extracted_json: dict
) -> str:
  model = genai.GenerativeModel("gemini-3.5-flash-lite")
  img = Image.open(image_path)
  prompt = LEVEL2_PROMPT_TEMPLATE.format(
      extracted_json=json.dumps(extracted_json), question=question
  )
  response = model.generate_content([prompt, img])
  return response.text.strip()