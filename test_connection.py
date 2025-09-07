"""
Test script to verify Google Sheets connection and configuration
Run this script to test your setup before running the full dashboard
"""

import os
import sys
from google_sheets_connector import GoogleSheetsConnector
from config import SHEET_ID, WORKSHEET_NAME, SHEET_NAME

def test_connection():
    """
    Test the Google Sheets connection and display basic information.
    
    WHY: It's important to verify the connection works before running the full dashboard.
    This helps identify configuration issues early.
    
    HOW: We test each component step by step and provide clear feedback about what's working.
    """
    print("🔍 Testing Google Sheets Connection...")
    print("=" * 50)
    
    # Check if credentials file exists
    credentials_path = "service_account_credentials.json"
    if not os.path.exists(credentials_path):
        print("❌ ERROR: Service account credentials file not found!")
        print(f"   Expected file: {credentials_path}")
        print("   Please download your service account JSON file and place it in the project directory.")
        return False
    
    print("✅ Service account credentials file found")
    
    # Test authentication
    try:
        print("🔐 Testing authentication...")
        connector = GoogleSheetsConnector(credentials_path)
        print("✅ Authentication successful")
    except Exception as e:
        print(f"❌ Authentication failed: {str(e)}")
        print("   Please check your service account JSON file and Google Cloud project setup.")
        return False
    
    # Test sheet access
    try:
        print("📊 Testing sheet access...")
        print(f"   Sheet ID: {SHEET_ID}")
        print(f"   Worksheet: {WORKSHEET_NAME}")
        
        # Get sheet info
        sheet_info = connector.get_sheet_info(SHEET_ID)
        if sheet_info:
            print(f"✅ Sheet found: {sheet_info['title']}")
            print(f"   Available worksheets: {[ws['title'] for ws in sheet_info['worksheets']]}")
        else:
            print("❌ Could not retrieve sheet information")
            return False
        
        # Test data fetching
        print("📥 Testing data fetch...")
        df = connector.get_sheet_data(SHEET_ID, WORKSHEET_NAME)
        
        if df is not None:
            print(f"✅ Data fetched successfully!")
            print(f"   Rows: {len(df)}")
            print(f"   Columns: {len(df.columns)}")
            print(f"   Column names: {list(df.columns)}")
            
            # Show first few rows
            print("\n📋 Sample data (first 3 rows):")
            print(df.head(3).to_string())
            
        else:
            print("❌ Could not fetch data from the sheet")
            return False
            
    except Exception as e:
        print(f"❌ Sheet access failed: {str(e)}")
        print("   Please check:")
        print("   1. Your SHEET_ID in config.py")
        print("   2. Your service account has access to the sheet")
        print("   3. The worksheet name is correct")
        return False
    
    # Test configuration
    print("\n⚙️ Testing configuration...")
    from config import COLUMNS
    
    missing_columns = []
    for col_name, col_value in COLUMNS.items():
        if col_value not in df.columns:
            missing_columns.append(f"{col_name} ('{col_value}')")
    
    if missing_columns:
        print("⚠️  WARNING: Some configured columns are missing:")
        for col in missing_columns:
            print(f"   - {col}")
        print("   Please update config.py with the correct column names.")
        print(f"   Available columns: {list(df.columns)}")
    else:
        print("✅ All configured columns found in the data")
    
    print("\n" + "=" * 50)
    print("🎉 Connection test completed!")
    
    if missing_columns:
        print("⚠️  Please update your configuration before running the dashboard.")
        return False
    else:
        print("✅ Everything looks good! You can now run the dashboard with:")
        print("   streamlit run dashboard.py")
        return True

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
