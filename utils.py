import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import tempfile

def get_invoices_from_db():
    """Retrieve invoice data from the SQLite database as a DataFrame."""
    conn = sqlite3.connect("invoices.db")
    df = pd.read_sql_query("SELECT * FROM invoices ORDER BY id", conn)
    conn.close()
    return df

def spending_trends(silent=False):
    """
    Generates charts based on invoice data from the database.
    If silent=True, returns file paths for PDF generation; otherwise, displays charts.
    """
    invoices_df = get_invoices_from_db()
    
    # Clean and convert date strings (assuming DD/MM/YYYY format)
    invoices_df['date'] = invoices_df['date'].astype(str)
    invoices_df['date'] = pd.to_datetime(invoices_df['date'], format='%d/%m/%Y', errors='coerce')
    
    # Convert total_amount to numeric and drop invalid rows
    invoices_df['total_amount'] = pd.to_numeric(invoices_df['total_amount'], errors='coerce')
    invoices_df = invoices_df.dropna(subset=['total_amount'])

    # Pie Chart: Spending by Category
    pie_chart_path = tempfile.mkstemp(suffix=".png")[1]
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    spending_by_category = invoices_df.groupby('category')['total_amount'].sum().reset_index()
    ax1.pie(spending_by_category['total_amount'],
            labels=spending_by_category['category'],
            autopct='%1.1f%%',
            startangle=140)
    ax1.set_title("Spending by Category")
    fig1.savefig(pie_chart_path, format="png")
    plt.close(fig1)

    # Line Chart: Spending Trends Over Time
    line_chart_path = tempfile.mkstemp(suffix=".png")[1]
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    spending_by_date = invoices_df.groupby('date')['total_amount'].sum().reset_index()
    spending_by_date = spending_by_date.dropna(subset=['date']).sort_values(by='date')
    ax2.plot(spending_by_date['date'], spending_by_date['total_amount'], marker='o')
    ax2.set_title("Spending Trends Over Time")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Total Amount (₹)")
    plt.xticks(rotation=45)
    fig2.tight_layout()
    fig2.savefig(line_chart_path, format="png")
    plt.close(fig2)

    # Bar Chart: Spending by Store
    bar_chart_path = tempfile.mkstemp(suffix=".png")[1]
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    spending_by_store = invoices_df.groupby('store_name')['total_amount'].sum().reset_index()
    sns.barplot(x='store_name', y='total_amount', data=spending_by_store, ax=ax3, palette='viridis')
    ax3.set_title("Spending by Store")
    plt.xticks(rotation=45)
    fig3.tight_layout()
    fig3.savefig(bar_chart_path, format="png")
    plt.close(fig3)

    if silent:
        return pie_chart_path, line_chart_path, bar_chart_path

    # Otherwise, display the charts in the UI
    st.image(pie_chart_path, caption="Spending by Category")
    st.image(line_chart_path, caption="Spending Trends Over Time")
    st.image(bar_chart_path, caption="Spending by Store")
