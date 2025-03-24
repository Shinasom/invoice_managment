import io
import json
import os
import re
import sqlite3
from fpdf import FPDF
from PIL import Image
from pdf2image import convert_from_bytes
from fuzzywuzzy import fuzz
import streamlit as st
import google.generativeai as genai
from google.cloud import vision
from google.oauth2 import service_account
import tempfile
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import spending_trends,get_invoices_from_db  # For chart generation (silent=True in PDF)

# -----------------------
# Streamlit Page Config
# -----------------------
st.set_page_config(page_title="Home", page_icon="🏠")

# -----------------------
# Google Gemini & Vision Setup
# -----------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
credentials = service_account.Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"])
client = vision.ImageAnnotatorClient(credentials=credentials)

# Required columns
REQUIRED_COLUMNS = ["store_name", "date", "bill_no", "total_amount", "extracted_text", "category", "gstin"]

# -----------------------
# Directory for Invoice Images
# -----------------------
SAVE_DIR = "invoice_images"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# -----------------------
# Database Functions
# -----------------------
def init_db():
    conn = sqlite3.connect("invoices.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS invoices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            store_name TEXT,
            date TEXT,
            bill_no TEXT,
            total_amount REAL,
            extracted_text TEXT,
            category TEXT,
            gstin TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_to_database(invoice_data, image):
    """
    Save invoice details to SQLite database and store the image in SAVE_DIR.
    Returns the invoice_id of the newly inserted record.
    """
    conn = sqlite3.connect("invoices.db")
    cursor = conn.cursor()
    try:
        total_amount = float(invoice_data.get("total_amount", 0))
    except ValueError:
        total_amount = 0.0

    cursor.execute("""
        INSERT INTO invoices (store_name, date, bill_no, total_amount, extracted_text, category, gstin)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        invoice_data.get("store_name", "Unknown"),
        invoice_data.get("date", "Unknown"),
        invoice_data.get("bill_no", "Unknown"),
        total_amount,
        invoice_data.get("extracted_text", ""),
        invoice_data.get("category", "Unknown"),
        invoice_data.get("gstin", "Unknown")
    ))
    invoice_id = cursor.lastrowid
    conn.commit()
    conn.close()

    # Save image to SAVE_DIR
    image_path = os.path.join(SAVE_DIR, f"{invoice_id}.png")
    image.save(image_path)
    return invoice_id

def clear_database():
    """Clear all invoices from the database, delete saved images, and reset invoice ID."""
    conn = sqlite3.connect("invoices.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM invoices")  # Delete all records
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='invoices'")  # Reset AUTOINCREMENT counter
    conn.commit()
    conn.close()

    # Remove all saved invoice images
    for filename in os.listdir(SAVE_DIR):
        file_path = os.path.join(SAVE_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Initialize DB
init_db()

# -----------------------
# Helper Functions
# -----------------------
def extract_text(image):
    """Extract text using Google Vision API."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    image_for_api = vision.Image(content=img_byte_arr)
    response = client.text_detection(image=image_for_api)
    texts = response.text_annotations
    return texts[0].description if texts else ""

def extract_entities(text):
    """Extract structured invoice data (including GSTIN, category) using Gemini."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    Extract the following details from this invoice text:
    - Store Name
    - Date (if the date is in a different format, convert it to DD/MM/YYYY format)
    - Bill Number
    - Total Amount
    - Category (choose from: Food, Travel, Office Supplies, Utilities, Others)
    - GSTIN

    Use the following **keywords for category classification**:
    - **Food**: restaurant, cafe, grocery, food, beverage, bakery, supermarket
    - **Travel**: flight, airline, hotel, taxi, fuel, petrol, Uber, Ola, bus, train
    - **Office Supplies**: stationery, printer, ink, paper, pen, laptop, computer, mouse, keyboard
    - **Utilities**: electricity, water, internet, mobile bill, phone bill, broadband, gas
    - **Others**: (use this if no relevant category is found)

    Provide ONLY a JSON response with these keys: "store_name", "date", "bill_no", "total_amount", "category", "gstin".
    Do NOT include any additional text, explanation, or formatting outside of the JSON object.

    Invoice text:
    {text}
    """
    response = model.generate_content(prompt)
    try:
        json_text = response.text.strip()
        json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
        invoice_data = json.loads(json_text)
        return invoice_data
    except json.JSONDecodeError:
        return {
            "store_name": "N/A",
            "date": "N/A",
            "bill_no": "N/A",
            "total_amount": "0",
            "category": "Others",
            "gstin": "N/A"
        }

def check_duplicate(extracted_text, threshold=90):
    """Check for duplicate invoices in the database using fuzzy matching."""
    df = get_invoices_from_db()
    for _, row in df.iterrows():
        stored_text = row["extracted_text"]
        similarity_score = fuzz.ratio(extracted_text, stored_text)
        if similarity_score >= threshold:
            return row["id"], similarity_score
    return None, 0

def process_pdf(uploaded_file):
    """Convert PDF pages to images and extract text."""
    images = convert_from_bytes(uploaded_file.read())
    extracted_text = extract_text(images[0])
    return extracted_text, images[0]

def calculate_total_amount():
    """Sum total_amount from database invoices."""
    df = get_invoices_from_db()
    df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce")
    return df["total_amount"].sum()

def wrap_text(text, max_width, pdf):
    """Wrap text so each line is <= max_width in PDF context."""
    words = text.split(" ")
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        if pdf.get_string_width(test_line) <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return "\n".join(lines)

# -----------------------
# PDF Generation
# -----------------------
def generate_invoice_pdf():
    """Generate a PDF with invoice summaries and charts."""
    progress_bar = st.progress(0)
    progress = 0

    # Step 1: PDF Setup & Summary Table
    progress += 10
    progress_bar.progress(min(progress, 100))
    time.sleep(0.5)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(200, 10, "Invoice Summary", ln=True, align="C")
    pdf.ln(10)
    
    total_sum = calculate_total_amount()
    
    pdf.set_font("Arial", style='B', size=10)
    col_widths = [15, 50, 42, 25, 30, 30]
    headers = ["Bill ID", "Store Name", "GSTIN", "Date", "Category", "Total Amount"]

    progress += 10
    progress_bar.progress(min(progress, 100))
    time.sleep(0.5)
    
    # Table Headers
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, border=1, align="C")
    pdf.ln()
    
    pdf.set_font("Arial", size=10)
    line_height = 5
    df = get_invoices_from_db()

    for _, invoice in df.iterrows():
        row_data = [
            str(invoice.get("id") or "N/A"),
            invoice.get("store_name") or "N/A",
            invoice.get("gstin") or "N/A",
            invoice.get("date") or "N/A",
            invoice.get("category") or "N/A",
            f"{float(invoice.get('total_amount') or 0):.2f}"
        ]
        wrapped_cells = []
        max_lines = 1
        for i, cell in enumerate(row_data):
            wrapped = wrap_text(cell, col_widths[i] - 2, pdf)
            lines = wrapped.split("\n")
            wrapped_cells.append(wrapped)
            if len(lines) > max_lines:
                max_lines = len(lines)
        row_height = line_height * max_lines
        x_start = pdf.get_x()
        y_start = pdf.get_y()
        for i, cell in enumerate(wrapped_cells):
            x_current = pdf.get_x()
            pdf.multi_cell(col_widths[i], line_height, cell, border=0)
            pdf.set_xy(x_current, y_start)
            pdf.rect(x_current, y_start, col_widths[i], row_height)
            pdf.set_xy(x_current + col_widths[i], y_start)
        pdf.ln(row_height)

        progress += 5
        progress_bar.progress(min(progress, 100))
    
    pdf.set_font("Arial", style='B', size=10)
    pdf.cell(sum(col_widths[:-1]), 8, "Total Amount", border=1, align="R")
    pdf.cell(col_widths[-1], 8, f"{total_sum:.2f}", border=1, align="C")
    pdf.ln(10)
    
    # Step 2: Add Spending Trends Charts
    pie_chart_path, line_chart_path, bar_chart_path = spending_trends(silent=True)

    df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce")
    spending_by_category = df.groupby("category")["total_amount"].sum().reset_index()
    
    pie_chart_path = tempfile.mkstemp(suffix=".png")[1]
    plt.figure(figsize=(8, 6))
    plt.pie(spending_by_category["total_amount"], labels=spending_by_category["category"], autopct='%1.1f%%', startangle=140)
    plt.title("Spending by Category")
    plt.savefig(pie_chart_path, format="png")
    plt.close()
    
    line_chart_path = tempfile.mkstemp(suffix=".png")[1]
    spending_by_date = df.groupby("date")["total_amount"].sum().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(spending_by_date["date"], spending_by_date["total_amount"], marker='o')
    plt.title("Spending Trends Over Time")
    plt.xlabel("Date")
    plt.ylabel("Total Amount (₹)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig(line_chart_path, format="png")
    plt.close()
    
    bar_chart_path = tempfile.mkstemp(suffix=".png")[1]
    spending_by_store = df.groupby("store_name")["total_amount"].sum().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x="store_name", y="total_amount", data=spending_by_store, palette="viridis")
    plt.title("Spending by Store")
    plt.xlabel("Store Name")
    plt.ylabel("Total Amount (₹)")
    plt.xticks(rotation=45)
    plt.savefig(bar_chart_path, format="png")
    plt.close()
    
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(200, 10, "Spending Trends", ln=True, align="C")
    pdf.ln(10)
    pdf.image(pie_chart_path, x=10, y=30, w=180)
    pdf.ln(100)
    pdf.image(line_chart_path, x=10, y=140, w=180)
    pdf.ln(100)
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=14)
    pdf.image(bar_chart_path, x=10, y=30, w=180)

    progress += 10
    progress_bar.progress(min(progress, 100))
    
    # Step 3: Add Invoice Images to the PDF
    for invoice_id, row in df.iterrows():
        pdf.add_page()
        temp_path = os.path.join(tempfile.gettempdir(), f"invoice_{row['id']}.png")
        image_path = os.path.join(SAVE_DIR, f"{row['id']}.png")
        try:
            pdf.image(image_path, x=10, y=10, w=180)
        except Exception as e:
            print(f"Error adding image for invoice {row['id']}: {e}")

        progress += 10
        progress_bar.progress(min(progress, 100))
    
    # Step 4: Save PDF and provide download button
    fd, pdf_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    pdf.output(pdf_path)

    progress = 100
    progress_bar.progress(progress)
    st.success("✅ Invoice Summary PDF generated successfully!")
    
    with open(pdf_path, "rb") as pdf_file:
        st.download_button(
            label="📄 Download Invoice Summary PDF", 
            data=pdf_file, 
            file_name="invoices_summary.pdf", 
            mime="application/pdf",
            key="download_invoice_summary"
        )
    
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

def display_invoice_details(invoice_id, invoice_data):
    store_name   = invoice_data.get("store_name") or "N/A"
    bill_no      = invoice_data.get("bill_no") or "N/A"
    date_value   = invoice_data.get("date") or "N/A"
    category     = invoice_data.get("category") or "N/A"
    total_amount = invoice_data.get("total_amount") or 0

    details = {
        "Field": ["Invoice ID", "Store Name", "Bill No", "Date", "Category", "Total Amount"],
        "Value": [
            invoice_id,
            store_name,
            bill_no,
            date_value,
            category,
            f"₹{float(total_amount):.2f}"
        ]
    }
    st.table(details)

def proceed_callback():
    invoice_data = extract_entities(st.session_state.duplicate_extracted_text)
    invoice_data["extracted_text"] = st.session_state.duplicate_extracted_text
    saved_id = save_to_database(invoice_data, st.session_state.duplicate_image)
    st.session_state["saved_invoice_id"] = saved_id
    st.session_state["saved_invoice_data"] = invoice_data
    st.session_state["show_table"] = True
    st.session_state["success_message"] = f"✅ Invoice Saved successfully! Invoice ID: {saved_id}"
    del st.session_state.duplicate_extracted_text
    del st.session_state.duplicate_image

def file_upload_handler(uploaded_file):
    if uploaded_file.type == "application/pdf":
        extracted_text, image = process_pdf(uploaded_file)
    else:
        image = Image.open(uploaded_file)
        extracted_text = extract_text(image)
    
    duplicate_id, similarity_score = check_duplicate(extracted_text)
    if duplicate_id:
        st.warning(f"⚠️ This invoice is similar to Invoice ID {duplicate_id} with a similarity score of {similarity_score}.")
        col1, col2 = st.columns(2)
        with col1:
            existing_image_path = os.path.join(SAVE_DIR, f"{duplicate_id}.png")
            st.image(existing_image_path, caption=f"Existing Invoice - ID: {duplicate_id}", use_container_width=True)
        with col2:
            st.image(image, caption="New Uploaded Invoice", use_container_width=True)
        st.session_state.duplicate_extracted_text = extracted_text
        st.session_state.duplicate_image = image
        st.button("Proceed to Save Anyway", key="proceed_duplicate_button", on_click=proceed_callback)
        return
    
    invoice_data = extract_entities(extracted_text)
    invoice_data["extracted_text"] = extracted_text
    invoice_id = save_to_database(invoice_data, image)
    st.session_state["saved_invoice_id"] = invoice_id
    st.session_state["saved_invoice_data"] = invoice_data
    st.session_state["show_table"] = True
    st.session_state["success_message"] = f"✅ Invoice Saved successfully! Invoice ID: {invoice_id}"

st.title("Invoice Management System")

if "file_upload_count" not in st.session_state:
    st.session_state.file_upload_count = 0

uploader_container = st.empty()
uploaded_file = uploader_container.file_uploader(
    "Upload Invoice Image or PDF",
    type=["png", "jpg", "jpeg", "pdf"],
    key=f"uploaded_file_{st.session_state.file_upload_count}"
)

if uploaded_file:
    file_upload_handler(uploaded_file)
    st.session_state.file_upload_count += 1
    uploader_container.empty()
    uploader_container.file_uploader(
        "Upload Invoice Image or PDF",
        type=["png", "jpg", "jpeg", "pdf"],
        key=f"uploaded_file_{st.session_state.file_upload_count}"
    )

success_placeholder = st.empty()
if st.session_state.get("success_message"):
    with success_placeholder:
        st.success(st.session_state["success_message"])
    del st.session_state["success_message"]

table_placeholder = st.empty()
if st.session_state.get("show_table", False):
    with table_placeholder:
        # Display only the invoice that was just saved using your display_invoice_details() function
        display_invoice_details(
            st.session_state["saved_invoice_id"],
            st.session_state["saved_invoice_data"]
        )
    st.session_state["show_table"] = False


if st.button("Sum of All Invoices"):
    total_sum = calculate_total_amount()
    st.success(f"💰 Total sum of all invoices: ₹{total_sum:.2f}")

if st.button("Clear All Data (Invoices & Images)"):
    clear_database()
    st.success("All invoices and saved images have been deleted.")

if st.button("Generate Invoice Summary PDF"):
    generate_invoice_pdf()
