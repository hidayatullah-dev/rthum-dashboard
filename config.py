"""
Configuration file for Google Sheets to Streamlit Dashboard
Replace the placeholder values with your actual Google Sheets information
"""

# Google Sheets Configuration
SHEET_NAME = "scrapping data version 2"  # Your Google Sheet name
WORKSHEET_NAME = "Scraping Data version 2"  # Your worksheet name
SHEET_ID = "1aJ8HQ83HSE1xbzwIulTo-EbgS10fcz6LAitt-nZ7pOk"  # Your Google Sheet ID

# Column Configuration - Replace with your actual column names
COLUMNS = {
    "date_column": "Member since",  # Column name for dates/timestamps
    "value_column": "Score",  # Column name for numerical values
    "category_column": "Category",  # Column name for categories/groups
    "name_column": "Job Title"  # Column name for names/labels
}

# Chart Configuration
CHART_TITLE = "Job Scraping Data Dashboard"  # Your desired chart title
X_AXIS_TITLE = "Categories"  # Your X-axis label
Y_AXIS_TITLE = "Scores"  # Your Y-axis label

# Refresh Configuration
REFRESH_INTERVAL = 3600  # Refresh interval in seconds (1 hour = 3600 seconds)
