import os
import sqlite3
import numpy as np
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
from fuzzywuzzy import fuzz
import pytesseract
import google.generativeai as genai
import json
import re


# Configure Gemini API Key
genai.configure(api_key="AIzaSyDUqJdPpP8E51hfJ-UhxDKqiJVgJau2j0E")





# Directory to save invoice images
SAVE_DIR = "saved_bills"
os.makedirs(SAVE_DIR, exist_ok=True)

# Required database columns
REQUIRED_COLUMNS = {
    "store_name": "TEXT",
    "date": "TEXT",
    "bill_no": "TEXT",
    "total_amount": "TEXT",
    "extracted_text": "TEXT"
}

def add_missing_columns(db_name="invoices.db"):
    """Ensure all required columns exist in the invoices table."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS invoices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            store_name TEXT,
            date TEXT,
            bill_no TEXT,
            total_amount TEXT,
            extracted_text TEXT
        )
    """)
    conn.commit()
    conn.close()

# Ensure database schema is correct
add_missing_columns()

def extract_text(image):
    """Extract text using Tesseract OCR."""
    return pytesseract.image_to_string(image, lang="eng")

import json
import re

def extract_entities(text):
    """Extract structured invoice data using Gemini 1.5 Flash API."""
    
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
    Extract the following details from this invoice text:
    - Store Name
    - Date
    - Bill Number
    - Total Amount
    Provide the response **strictly in JSON format** with keys: "store_name", "date", "bill_no", "total_amount".
    
    Invoice text:
    {text}
    """

    response = model.generate_content(prompt)

    try:
        # Clean extra markdown or incorrect formatting
        clean_text = re.sub(r"```json\n(.*?)\n```", r"\1", response.text, flags=re.DOTALL).strip()
        
        # Parse JSON
        extracted_data = json.loads(clean_text)
    except json.JSONDecodeError:
        extracted_data = {"error": "Failed to parse response. Raw output: " + response.text}

    return extracted_data



def check_duplicate(extracted_text, db_name="invoices.db", threshold=90):
    """Compare extracted text with database records for duplicate detection."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT id, extracted_text FROM invoices")
    existing_texts = cursor.fetchall()
    conn.close()
    
    for stored_id, stored_text in existing_texts:
        similarity_score = fuzz.ratio(extracted_text, stored_text)
        if similarity_score >= threshold:
            return stored_id, similarity_score
    
    return None, 0

def save_to_database(invoice_data, image):
    """Save invoice details to SQLite database and store the image."""
    conn = sqlite3.connect("invoices.db")
    cursor = conn.cursor()
    
    # Ensure total_amount is a float (default to 0 if invalid)
    try:
        total_amount = float(invoice_data.get("total_amount", 0))
    except ValueError:
        total_amount = 0.0

    cursor.execute("""
        INSERT INTO invoices (store_name, date, bill_no, total_amount, extracted_text)
        VALUES (?, ?, ?, ?, ?)
    """, (invoice_data.get("store_name", "Unknown"), invoice_data.get("date", "Unknown"),
          invoice_data.get("bill_no", "Unknown"), total_amount,
          invoice_data.get("extracted_text", "")))

    invoice_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    image_path = os.path.join(SAVE_DIR, f"{invoice_id}.png")
    image.save(image_path)
    return invoice_id

def process_pdf(uploaded_file):
    """Convert PDF pages to images and extract text."""
    images = convert_from_bytes(uploaded_file.read())
    extracted_text = extract_text(images[0])
    return extracted_text, images[0]

# Streamlit UI
st.title("Invoice OCR Scanner")

uploaded_file = st.file_uploader("Upload Invoice Image or PDF", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type in ["png", "jpg", "jpeg"]:
        image = Image.open(uploaded_file)
        extracted_text = extract_text(image)
    elif file_type == "pdf":
        extracted_text, image = process_pdf(uploaded_file)
    
    invoice_data = extract_entities(extracted_text)
    invoice_data["extracted_text"] = extracted_text
    
    st.subheader("Extracted Invoice Details")
    st.json(invoice_data)
    
    duplicate_id, similarity_score = check_duplicate(extracted_text)
    
    if duplicate_id:
        st.warning(f"⚠ Duplicate Invoice Detected! (ID: {duplicate_id}, Similarity Score: {similarity_score}%)")
        existing_image_path = os.path.join(SAVE_DIR, f"{duplicate_id}.png")
        if os.path.exists(existing_image_path):
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="New Uploaded Invoice", use_column_width=True)
            with col2:
                st.image(existing_image_path, caption="Existing Invoice", use_column_width=True)
        
        if st.button("Proceed to Save Anyway"):
            saved_id = save_to_database(invoice_data, image)
            st.success(f"Invoice saved with ID: {saved_id}.")
    else:
        saved_id = save_to_database(invoice_data, image)
        st.success(f"✅ Invoice saved with ID: {saved_id}.")
