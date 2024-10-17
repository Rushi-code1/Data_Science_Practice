import streamlit as st
import pandas as pd
from datetime import datetime

class FinancialReportUI:
    def __init__(self):
        self.data = None

    def upload_file(self):
        uploaded_file = st.file_uploader("Upload your financial dataset (CSV/Excel)", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                self.data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                self.data = pd.read_excel(uploaded_file)
            
            # Ensure 'Date' column is in datetime format
            if 'Date' in self.data.columns:
                self.data['Date'] = pd.to_datetime(self.data['Date'], errors='coerce')
            st.success("File uploaded successfully!")

    def select_date_range(self):
        if self.data is not None:
            st.subheader("Select Date Range")
            min_date = self.data['Date'].min()  # Get the minimum date
            max_date = self.data['Date'].max()  # Get the maximum date

            # Ensure min_date and max_date are valid
            if pd.notna(min_date) and pd.notna(max_date):
                start_date = st.date_input("Start date", value=min_date.date(), min_value=min_date.date(), max_value=max_date.date())
                end_date = st.date_input("End date", value=max_date.date(), min_value=min_date.date(), max_value=max_date.date())
                return start_date, end_date
            else:
                st.warning("No valid dates available in the uploaded dataset.")
        else:
            st.warning("Please upload a dataset first.")

    def generate_report(self):
        if self.data is not None:
            start_date, end_date = self.select_date_range()
            filtered_data = self.data[(self.data['Date'] >= pd.Timestamp(start_date)) & (self.data['Date'] <= pd.Timestamp(end_date))]
            if not filtered_data.empty:
                st.subheader("Financial Report")
                st.write(filtered_data)  # Display the filtered report
                # Additional calculations can be done here (e.g., Total Revenue, Expenses)
                total_revenue = filtered_data['Revenue'].sum()
                total_expenses = filtered_data['Expenses'].sum()
                net_profit = total_revenue - total_expenses
                st.write(f"Total Revenue: ${total_revenue:.2f}")
                st.write(f"Total Expenses: ${total_expenses:.2f}")
                st.write(f"Net Profit: ${net_profit:.2f}")
            else:
                st.warning("No data available for the selected date range.")

    def run(self):
        st.title("Financial Report Generation")
        self.upload_file()
        if self.data is not None:
            self.generate_report()

# To run the UI
if __name__ == "__main__":
    ui = FinancialReportUI()
    ui.run()
