import streamlit as st 
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
import google.generativeai as genai
from utils import spending_trends, get_invoices_from_db  # Import functions from utils

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

# Custom CSS for full-width layout
st.markdown(
    """
    <style>
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Fetch invoice data from the database
df = get_invoices_from_db()

# -----------------------
# KPI Cards
# -----------------------
if not df.empty:
    # Ensure total_amount is numeric
    df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce")
    
    # Calculate key metrics
    total_invoices = len(df)
    total_spending = df["total_amount"].sum()
    average_invoice = total_spending / total_invoices if total_invoices else 0

    # Calculate highest expense category
    category_totals = df.groupby("category")["total_amount"].sum().reset_index()
    if not category_totals.empty:
        highest = category_totals.loc[category_totals["total_amount"].idxmax()]
        highest_category = highest["category"]
        highest_amount = highest["total_amount"]
    else:
        highest_category = "N/A"
        highest_amount = 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Invoices", total_invoices)
    col2.metric("Total Spending", f"₹{total_spending:,.2f}")
    col3.metric("Average Invoice", f"₹{average_invoice:,.2f}")
    col4.metric("Highest Expense Category", f"{highest_category} (₹{highest_amount:,.2f})")
else:
    st.info("No invoice data available to compute KPIs.")

# -----------------------
# Create Tabs for Charts, Tables, and AI Insights
# -----------------------
tab1, tab2, tab3 = st.tabs(["📈 Charts", "📋 Tables", "🤖 AI Insights"])

with tab1:
    st.subheader("Spending Trends")
    if not df.empty:
        spending_trends()  # spending_trends() in utils now uses get_invoices_from_db() internally
    else:
        st.warning("No invoice data available. Upload invoices to view spending trends.")

with tab2:
    st.subheader("Interactive Invoice Data")
    if not df.empty:
        # Use the DataFrame from the database
        columns_order = ["id", "store_name", "gstin", "date", "category", "total_amount"]
        df_display = df[columns_order].copy()
        
        # Rename columns for display
        df_display = df_display.rename(columns={
            "id": "Bill ID",
            "store_name": "Store Name",
            "gstin": "GSTIN",
            "date": "Date",
            "category": "Category",
            "total_amount": "Total Amount"
        })
        
        # Ensure "Total Amount" is numeric
        df_display["Total Amount"] = pd.to_numeric(df_display["Total Amount"], errors="coerce")
        
        gb = GridOptionsBuilder.from_dataframe(df_display)
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_default_column(editable=False, filter=True, sortable=True)
        
        # Set specific column widths
        gb.configure_column("Bill ID", width=100)
        gb.configure_column("Store Name", width=250)
        gb.configure_column("GSTIN", width=200)
        gb.configure_column("Date", width=100)
        gb.configure_column("Category", width=120)
        gb.configure_column("Total Amount", 
                            type=["numericColumn", "numberColumnFilter", "customNumericFormat"],
                            custom_format_string="₹0,0.00",
                            width=150)
        
        gridOptions = gb.build()
        gridOptions["domLayout"] = "autoHeight"
        
        AgGrid(df_display, gridOptions=gridOptions, height=500, fit_columns_on_grid_load=True)
    else:
        st.info("No invoice data available.")

with tab3:
    st.subheader("AI Insights")
    
    def generate_ai_insights(invoices_df):
        if invoices_df.empty:
            return "No data available for insights."
        
        invoice_text = invoices_df.to_string(index=False)
        
        # Revised prompt: generate insights and actionable recommendations for the client.
        prompt = (
            "Analyze the following invoice data and provide 5 to 10 bullet-point insights along with actionable recommendations for the client. "
            "Focus on identifying key spending trends, anomalies, and cost drivers, and include only those insights that are directly useful for "
            "making financial decisions (e.g., high spending areas, opportunities for vendor negotiation, unusual spikes in costs). "
            "Exclude suggestions about internal data standardization, invoice formatting issues, or unclear notations unless they significantly impact "
            "the spending or payment process. Output only bullet points.\n\n"
            "Invoice Data:\n"
            f"{invoice_text}"
        )
        
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        return response.text.strip()
    
    insights = generate_ai_insights(df)
    st.markdown(insights)
