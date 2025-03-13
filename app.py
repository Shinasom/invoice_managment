import os
import sqlite3
import re
import numpy as np
import cv2
import pytesseract
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
from fuzzywuzzy import fuzz

# Set up Tesseract path (Update for your system)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Directory to save images
SAVE_DIR = "saved_bills"
os.makedirs(SAVE_DIR, exist_ok=True)

# Poppler Path - Make sure it is correct
POPPLER_PATH = r"C:\Users\shina\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"

# Required database columns
REQUIRED_COLUMNS = {
    "store_name": "TEXT",
    "date": "TEXT",
    "bill_no": "TEXT",
    "total_amount": "TEXT",
    "extracted_text": "TEXT"
}

def get_existing_columns(db_name="invoices.db"):
    """Fetch the current column names from the invoices table."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(invoices)")
    existing_columns = {col[1] for col in cursor.fetchall()}
    conn.close()
    return existing_columns

def add_missing_columns(db_name="invoices.db"):
    """Ensure all required columns exist in the invoices table."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    existing_columns = get_existing_columns(db_name)
    missing_columns = {col: dtype for col, dtype in REQUIRED_COLUMNS.items() if col not in existing_columns}

    if missing_columns:
        for column, dtype in missing_columns.items():
            cursor.execute(f"ALTER TABLE invoices ADD COLUMN {column} {dtype}")
            print(f"✅ Added missing column: {column} ({dtype})")

        conn.commit()
    conn.close()

# Ensure database schema is correct
add_missing_columns()

def preprocess_image(image):
    """Convert image to grayscale and apply thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text(image):
    """Extract text using Tesseract OCR."""
    return pytesseract.image_to_string(image).lower()

def parse_invoice(text):
    """Extract structured data from the OCR text."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    store_name = lines[0] if lines else "Unknown"
    
    date_match = re.search(r'\d{2}[/.-]\d{2}[/.-]\d{4}', text)
    date = date_match.group() if date_match else "Not Found"
    
    bill_no_match = re.search(r'bill\s*no[:\s]*([\w-]+)', text, re.IGNORECASE)
    bill_no = bill_no_match.group(1) if bill_no_match else "Not Found"
    
    total_match = re.search(r'(?:total\s*amount|grand\s*total|total)\s*[:]?[\s$]*([\d,]+(?:\.\d{2})?)', text, re.IGNORECASE)
    total_amount = total_match.group(1) if total_match else "Not Found"

    return {
        "Store Name": store_name,
        "Date": date,
        "Bill No": bill_no,
        "Total Amount": total_amount,
        "Extracted Text": text
    }

def check_duplicate(extracted_text, db_name="invoices.db", threshold=90):
    """Compare extracted text with database records for duplicate detection."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT id, extracted_text FROM invoices")
    existing_texts = cursor.fetchall()

    for row in existing_texts:
        stored_id, stored_text = row
        similarity_score = fuzz.ratio(extracted_text, stored_text)
        if similarity_score >= threshold:
            conn.close()
            return stored_id, similarity_score

    conn.close()
    return None, 0

def save_to_database(invoice_data, image):
    """Save invoice details to SQLite database and store the image."""
    conn = sqlite3.connect("invoices.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS invoices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            store_name TEXT,
            date TEXT,
            bill_no TEXT,
            total_amount INTEGER,  -- Ensure INTEGER type
            extracted_text TEXT
        )
    """)

    # Convert total_amount to integer (remove commas if present)
    try:
        total_amount = int(invoice_data["Total Amount"].replace(",", ""))  # Ensure integer format
    except ValueError:
        total_amount = 0  # Default to 0 if conversion fails

    cursor.execute("""
        INSERT INTO invoices (store_name, date, bill_no, total_amount, extracted_text)
        VALUES (?, ?, ?, ?, ?)
    """, (invoice_data["Store Name"], invoice_data["Date"], invoice_data["Bill No"],
          total_amount, invoice_data["Extracted Text"]))

    invoice_id = cursor.lastrowid
    conn.commit()
    conn.close()

    image_path = os.path.join(SAVE_DIR, f"{invoice_id}.png")
    image.save(image_path)

    return invoice_id


def process_pdf(uploaded_file):
    """Convert PDF pages to images and extract text."""
    images = convert_from_bytes(uploaded_file.read(), poppler_path=POPPLER_PATH)
    extracted_text = ""

    for img in images:
        img_np = np.array(img)
        processed_img = preprocess_image(img_np)
        extracted_text += extract_text(processed_img) + "\n"

    return extracted_text, images[0]

# Streamlit UI
st.title("Invoice OCR Scanner")

uploaded_file = st.file_uploader("Upload Invoice Image or PDF", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type in ["png", "jpg", "jpeg"]:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        processed_image = preprocess_image(image_np)
        extracted_text = extract_text(processed_image)

    elif file_type == "pdf":
        extracted_text, image = process_pdf(uploaded_file)

    invoice_data = parse_invoice(extracted_text)

    st.subheader("Extracted Invoice Details")
    st.json(invoice_data)

    # Check for duplicates
    duplicate_id, similarity_score = check_duplicate(invoice_data["Extracted Text"])

    if duplicate_id:
        st.warning(f"⚠ Duplicate Invoice Detected! (ID: {duplicate_id}, Similarity Score: {similarity_score}%)")

        # Load and display the existing stored invoice
        existing_image_path = os.path.join(SAVE_DIR, f"{duplicate_id}.png")
        if os.path.exists(existing_image_path):
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="New Uploaded Invoice", use_column_width=True)
            with col2:
                st.image(existing_image_path, caption="Existing Invoice in Database", use_column_width=True)
        else:
            st.error("Stored invoice image not found for comparison.")

        # Allow user to save duplicate invoice anyway
        if st.button("Proceed to Save Anyway"):
            saved_id = save_to_database(invoice_data, image)
            st.success(f"Invoice saved with ID: {saved_id}. Image stored successfully.")
    else:
        saved_id = save_to_database(invoice_data, image)
        st.success(f"✅ Invoice saved with ID: {saved_id}. Image stored successfully.")
