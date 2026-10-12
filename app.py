import json
import os
from bill_extractor import answer_bill_question, extract_bill_level1
from PIL import Image
import streamlit as st

st.set_page_config(page_title="WTQ Utility Bill Decoder", layout="wide")
st.title("AI Utility Bill Decoder & Investigator")
st.caption(
    "Automated Bill Extraction & Customer Q&A Powered by Google"
    " gemini-3.5-flash-lite"
)

col1, col2 = st.columns([1, 1])

with col1:
  st.subheader("1. Bill Ingestion")
  uploaded_file = st.file_uploader(
      "Choose a bill image (KE, LESCO, IESCO)", type=["png", "jpg", "jpeg"]
  )

  if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Electricity Bill", use_container_width=True)

    # Temporary save for API processing
    temp_path = "temp_bill.png"
    image.save(temp_path)

    if st.button("Extract Bill Details (Level 1)", type="primary"):
      with st.spinner("Extracting billing data with gemini-3.5-flash-lite..."):
        data = extract_bill_level1(temp_path)
        st.session_state["extracted_data"] = data
        st.success("Extraction Complete!")

with col2:
  if "extracted_data" in st.session_state:
    data = st.session_state["extracted_data"]
    st.subheader("2. Level 1: Extracted JSON")

    # Metrics summary
    m1, m2, m3 = st.columns(3)
    m1.metric("Provider", data.get("provider", "N/A"))
    m2.metric("Units Consumed", data.get("units_consumed", "N/A"))
    m3.metric("Payable Amount", f"PKR {data.get('payable_within_due_date', 0)}")

    st.json(data)

    st.subheader("3. Level 2: Customer Query Assistant")
    sample_q = st.selectbox(
        "Select or type a question:",
        [
            "How much of my bill is taxes?",
            "How many months in my history went above 200 units?",
            "What is the breakdown of my electricity charges?",
            "Why is my bill higher than my actual unit consumption?",
            "What is the late payment surcharge penalty?",
        ],
    )
    custom_q = st.text_input("Or ask a custom question:", value=sample_q)

    if st.button("Ask Assistant"):
      with st.spinner("Analyzing bill..."):
        ans = answer_bill_question("temp_bill.png", custom_q, data)
        st.info(ans)