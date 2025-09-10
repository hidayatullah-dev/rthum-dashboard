"""
Google Sheets Connector Module
Handles authentication and data fetching from Google Sheets using service account
"""

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import streamlit as st
from typing import Optional, Dict, Any
import time
from datetime import datetime

class GoogleSheetsConnector:
    """
    A class to handle Google Sheets API connections and data operations.
    
    WHY: We use a class-based approach to encapsulate all Google Sheets operations,
    making the code more organized and reusable. This follows the Single Responsibility
    Principle where this class only handles Google Sheets operations.
    
    HOW: The class uses the Google Sheets API v4 with service account authentication,
    which is the most secure and reliable method for server-to-server communication.
    """
    
    def __init__(self, credentials_path: str = None, credentials: Credentials = None):
        """
        Initialize the Google Sheets connector with service account credentials.
        
        WHY: Service accounts are the recommended way for server applications to access
        Google APIs without user interaction. They provide secure, programmatic access.
        
        HOW: We load the service account credentials from a JSON file or use provided
        credentials and create the necessary API clients for both gspread (simpler operations) 
        and the Google Sheets API (more advanced operations).
        
        Args:
            credentials_path (str, optional): Path to the service account JSON credentials file
            credentials (Credentials, optional): Pre-loaded credentials object
        """
        self.credentials_path = credentials_path
        self.credentials = credentials
        self.service = None
        self.gc = None
        self._authenticate()
    
    @classmethod
    def from_credentials(cls, credentials: Credentials):
        """
        Create a GoogleSheetsConnector instance from pre-loaded credentials.
        
        WHY: This allows using credentials from Streamlit secrets or other sources
        without needing a file path.
        
        HOW: We create a new instance with the provided credentials object.
        
        Args:
            credentials (Credentials): Pre-loaded Google credentials
            
        Returns:
            GoogleSheetsConnector: New connector instance
        """
        instance = cls(credentials=credentials)
        return instance
    
    def _authenticate(self):
        """
        Authenticate with Google Sheets API using service account credentials.
        
        WHY: Authentication is required before we can access any Google Sheets data.
        We use OAuth2 service account flow which is ideal for automated applications.
        
        HOW: We create credentials from the service account JSON file or use provided
        credentials and use them to build both the Google Sheets API service and gspread client. 
        We define the scopes needed for reading and writing to Google Sheets.
        """
        try:
            # Define the scopes needed for Google Sheets access
            # WHY: Scopes determine what permissions our application has
            # HOW: We request read-only access to spreadsheets for security
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets.readonly',
                'https://www.googleapis.com/auth/drive.readonly'
            ]
            
            # Load service account credentials
            # WHY: Service account credentials can come from a file or be provided directly
            # HOW: We check if credentials are provided, otherwise load from file
            if self.credentials is not None:
                # Use provided credentials (from Streamlit secrets)
                credentials = self.credentials
                if hasattr(credentials, 'with_scopes'):
                    credentials = credentials.with_scopes(scopes)
            else:
                # Load from file (local development)
                credentials = Credentials.from_service_account_file(
                    self.credentials_path, 
                    scopes=scopes
                )
            
            # Create Google Sheets API service
            # WHY: The Google Sheets API service provides programmatic access to sheets
            # HOW: We use the discovery build method with our credentials
            self.service = build('sheets', 'v4', credentials=credentials)
            
            # Create gspread client for simpler operations
            # WHY: gspread provides a more Pythonic interface for common operations
            # HOW: We authenticate gspread with our credentials
            self.gc = gspread.authorize(credentials)
            
        except Exception as e:
            st.error(f"Authentication failed: {str(e)}")
            st.error("Please ensure your service account credentials are valid and have the correct permissions.")
            raise
    
    def get_sheet_data(self, sheet_id: str, worksheet_name: str = "Sheet1") -> Optional[pd.DataFrame]:
        """
        Fetch data from a Google Sheet and return it as a Pandas DataFrame.
        
        WHY: We need to convert Google Sheets data into a format that's easy to work
        with in Python. Pandas DataFrames are perfect for data manipulation and analysis.
        
        HOW: We use the Google Sheets API to get all values from the specified range,
        then convert the raw data into a Pandas DataFrame with proper column headers.
        
        Args:
            sheet_id (str): The Google Sheet ID (found in the URL)
            worksheet_name (str): The name of the worksheet to read from
            
        Returns:
            Optional[pd.DataFrame]: The sheet data as a DataFrame, or None if error
        """
        try:
            # Construct the range to read all data from the worksheet
            # WHY: We want to read all data, so we use the worksheet name without specific range
            # HOW: The range format is "WorksheetName!A:Z" where A:Z covers all columns
            range_name = f"{worksheet_name}!A:Z"
            
            # Call the Google Sheets API to get values
            # WHY: We use the spreadsheets.values.get method to retrieve data
            # HOW: We specify the spreadsheet ID and range, then call execute() to make the request
            result = self.service.spreadsheets().values().get(
                spreadsheetId=sheet_id,
                range=range_name
            ).execute()
            
            # Extract values from the API response
            # WHY: The API returns data in a specific format that we need to parse
            # HOW: We get the 'values' key from the result, which contains the actual data
            values = result.get('values', [])
            
            if not values:
                st.warning("No data found in the specified sheet.")
                return None
            
            # Convert to DataFrame
            # WHY: We need the first row as column headers and the rest as data
            # HOW: We use pandas.DataFrame constructor with the first row as columns
            df = pd.DataFrame(values[1:], columns=values[0])
            
            # Clean up the DataFrame
            # WHY: Google Sheets data might have empty rows or inconsistent formatting
            # HOW: We remove completely empty rows and strip whitespace from string columns
            df = df.dropna(how='all')  # Remove rows where all values are NaN
            df = df.map(lambda x: x.strip() if isinstance(x, str) else x)  # Strip whitespace
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching data from Google Sheets: {str(e)}")
            return None
    
    def get_sheet_info(self, sheet_id: str) -> Optional[Dict[str, Any]]:
        """
        Get basic information about the Google Sheet.
        
        WHY: It's useful to know what worksheets are available and basic sheet metadata.
        This helps with debugging and provides user feedback.
        
        HOW: We use the spreadsheets.get method to retrieve sheet metadata including
        worksheet names, dimensions, and other properties.
        
        Args:
            sheet_id (str): The Google Sheet ID
            
        Returns:
            Optional[Dict[str, Any]]: Sheet information including worksheet names
        """
        try:
            # Get spreadsheet metadata
            # WHY: We need to know what worksheets are available in the sheet
            # HOW: We use the spreadsheets.get method to get detailed information
            sheet_metadata = self.service.spreadsheets().get(
                spreadsheetId=sheet_id
            ).execute()
            
            # Extract worksheet information
            # WHY: We want to show users what worksheets are available
            # HOW: We iterate through the sheets property and extract names and dimensions
            worksheets = []
            for sheet in sheet_metadata.get('sheets', []):
                worksheet_info = {
                    'title': sheet['properties']['title'],
                    'sheet_id': sheet['properties']['sheetId'],
                    'row_count': sheet['properties']['gridProperties']['rowCount'],
                    'column_count': sheet['properties']['gridProperties']['columnCount']
                }
                worksheets.append(worksheet_info)
            
            return {
                'title': sheet_metadata.get('properties', {}).get('title', 'Unknown'),
                'worksheets': worksheets
            }
            
        except Exception as e:
            st.error(f"Error getting sheet info: {str(e)}")
            return None
