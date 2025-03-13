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
import shutil
from fpdf import FPDF



# Configure Gemini API Key
genai.configure(api_key="AIzaSyDUqJdPpP8E51hfJ-UhxDKqiJVgJau2j0E")





SAVE_DIR = "saved_bills"
DB_NAME = "invoices.db"
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

def calculate_total_amount():
    """Calculate the sum of total_amount column from invoices.db."""
    conn = sqlite3.connect("invoices.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT SUM(total_amount) FROM invoices")
    total = cursor.fetchone()[0]  # Fetch the sum (None if no records)
    
    conn.close()
    
    return total if total is not None else 0.0


def clear_database():
    """Delete all rows from the invoices table without dropping the table."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM invoices")  # Clears all rows but keeps table
    conn.commit()
    conn.close()

def clear_saved_images():
    """Delete all images inside the saved_bills directory."""
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)  # Deletes the folder and all contents
        os.makedirs(SAVE_DIR, exist_ok=True)  # Recreate the folder




def generate_invoice_pdf():
    """Generate a PDF with an invoice summary table on the first page and invoice images on subsequent pages."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=14)
    
    # Title
    pdf.cell(200, 10, "Invoice Summary", ln=True, align="C")
    pdf.ln(10)

    # Connect to database and fetch invoice data
    conn = sqlite3.connect("invoices.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, store_name, date, total_amount FROM invoices")
    invoices = cursor.fetchall()

    # Calculate total amount
    total_sum = sum(row[3] for row in invoices)

    # Table Headers
    pdf.set_font("Arial", style='B', size=10)
    col_widths = [20, 60, 40, 30]
    headers = ["Bill ID", "Store Name", "Date", "Total Amount"]

    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, border=1, align="C")
    pdf.ln()

    # Table Data
    pdf.set_font("Arial", size=10)
    for row in invoices:
        pdf.cell(col_widths[0], 8, str(row[0]), border=1, align="C")
        pdf.cell(col_widths[1], 8, row[1], border=1)
        pdf.cell(col_widths[2], 8, row[2], border=1, align="C")
        pdf.cell(col_widths[3], 8, f"{row[3]:.2f}", border=1, align="C")
        pdf.ln()

    # Total sum row
    pdf.set_font("Arial", style='B', size=10)
    pdf.cell(sum(col_widths[:3]), 8, "Total Amount", border=1, align="R")
    pdf.cell(col_widths[3], 8, f"{total_sum:.2f}", border=1, align="C")
    pdf.ln(10)

    conn.close()

    # Add invoice images on new pages
    image_folder = "saved_bills"
    images = [f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".jpeg"))]

    if not images:
        st.warning("No invoice images found to generate PDF.")
        return

    for img in images:
        img_path = os.path.join(image_folder, img)
        pdf.add_page()
        pdf.image(img_path, x=10, y=10, w=180)

    # Save and offer PDF for download
    pdf_output_path = "invoices_summary.pdf"
    pdf.output(pdf_output_path)
    st.success("✅ Invoice Summary PDF generated successfully!")

    with open(pdf_output_path, "rb") as pdf_file:
        st.download_button(label="📄 Download Invoice Summary PDF", data=pdf_file, file_name="invoices_summary.pdf", mime="application/pdf")



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

if st.button("Sum of All Invoices"):
    total_sum = calculate_total_amount()
    st.success(f"💰 Total sum of all invoices: ₹{total_sum:.2f}")

if st.button("Clear All Data (Invoices & Images)"):
    clear_database()
    clear_saved_images()
    st.success("All invoices and saved images have been deleted.")
    st.rerun()

if st.button("Generate Invoice Summary PDF"):
    generate_invoice_pdf()
