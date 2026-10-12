```markdown
# WTQ Build Track — Utility Bill Decoder

An automated extraction and grounded question-answering system for Pakistani utility electricity bills (K-Electric, LESCO, IESCO). The solution processes raw bill images to output structured, single-line JSON representations (Level 1) and answers grounded consumer questions with exact arithmetic validation (Level 2).

---

## 1. Runtime & Environment
- **Runtime:** Python 3.14
- **Operating System:** Windows 10/11 (fully cross-platform compatible)
- **Model Used:** Google Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`)

---

## 2. Setup Steps & Dependencies

1. **Activate Virtual Environment:**
   ```powershell
   # Windows PowerShell
   .\.venv\Scripts\Activate

```

2. **Install Pinned Dependencies:**
Install dependencies specified in `requirements.txt`:
```bash
pip install -r requirements.txt

```


3. **Configure Environment Variables:**
Copy `.env.example` to `.env` and insert your Google AI Studio API key:
```env
MODEL_PROVIDER=google
MODEL_NAME=gemini-3.5-flash-lite
API_KEY=your_actual_api_key_here

```



---

## 3. Exact Commands to Run the Project

### A. Batch Processing (Generate Level 1 & Level 2 CSV Deliverables)

To process the 10 test bills and generate both evaluation CSV files:

```bash
python run_test_batch.py

```

* **Inputs Read:**
* `data/test/bills/` (10 test images)
* `data/test/csv/level1.csv` (Level 1 template)
* `data/test/csv/level2.csv` (Level 2 template)


* **Outputs Generated:**
* `output/level1.csv` (10 rows, compact single-line JSON per bill)
* `output/level2.csv` (120 rows, grounded answers per question)



### B. Interactive Dashboard & Verification

To run the analyst UI for live inspection and screen demonstrations:

```bash
streamlit run app.py

```

Access the application in your browser at `http://localhost:8501`.

---

## 4. AI Usage & Tools Disclosure

In accordance with Section 7.4 of the WTQ Competition Guide, all models, tools, and services utilized are listed below:

* **Approved Model:** Google Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`)


* **Model Provider API:** Google AI Studio (`google-generativeai==0.8.4`)
* **AI Coding Assistance:** Google Gemini
* **OCR Libraries:** Tesseract OCR via `pytesseract==0.3.13` (supplementary text extraction)
* **Third-Party Prebuilt Invoice/Extraction Services:** none
* **Other AI Tools / Services:** none

```

```