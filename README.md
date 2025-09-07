# Google Sheets Live Dashboard

A Python application that connects to Google Sheets using the Google Sheets API, pulls data into a Pandas DataFrame, and creates an interactive live visualization dashboard using Streamlit and Plotly.

## Features

- 🔄 **Live Data Sync**: Automatically refreshes data from Google Sheets
- 📊 **Interactive Charts**: Bar charts, line charts, and pie charts using Plotly
- 📋 **Data Table**: View and explore raw data with search and filtering
- ⚙️ **Easy Configuration**: Simple configuration file for customization
- 🔐 **Secure Authentication**: Uses Google service account for secure API access
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Prerequisites

Before you begin, ensure you have:

1. **Python 3.8 or higher** installed on your system
2. **A Google Cloud Project** with Google Sheets API enabled
3. **A Google Service Account** with access to your Google Sheet
4. **A Google Sheet** with data you want to visualize

## Setup Instructions

### Step 1: Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

**WHY**: We need specific versions of libraries to ensure compatibility and avoid conflicts.
**HOW**: The requirements.txt file lists all necessary packages with their versions.

### Step 2: Set Up Google Cloud Project

1. **Go to the Google Cloud Console**: https://console.cloud.google.com/
2. **Create a new project** or select an existing one
3. **Enable the Google Sheets API**:
   - Go to "APIs & Services" > "Library"
   - Search for "Google Sheets API"
   - Click on it and press "Enable"
4. **Enable the Google Drive API** (required for sheet access):
   - Search for "Google Drive API"
   - Click on it and press "Enable"

**WHY**: Google Sheets API requires proper authentication and permissions to access your data.
**HOW**: We enable the necessary APIs in Google Cloud Console to grant our application access.

### Step 3: Create Service Account

1. **Go to "APIs & Services" > "Credentials"**
2. **Click "Create Credentials" > "Service Account"**
3. **Fill in the service account details**:
   - Name: `sheets-dashboard-service`
   - Description: `Service account for Google Sheets dashboard`
4. **Click "Create and Continue"**
5. **Skip the "Grant access" step** (click "Done")
6. **Click on your newly created service account**
7. **Go to the "Keys" tab**
8. **Click "Add Key" > "Create new key"**
9. **Choose "JSON" format and download the file**
10. **Rename the downloaded file to `service_account_credentials.json`**
11. **Place it in your project directory**

**WHY**: Service accounts provide secure, programmatic access to Google APIs without requiring user login.
**HOW**: We create a service account and download its credentials in JSON format for authentication.

### Step 4: Share Your Google Sheet

1. **Open your Google Sheet**
2. **Click the "Share" button** (top right)
3. **Add the service account email** (found in your JSON file as `client_email`)
4. **Give it "Editor" or "Viewer" permissions** (depending on your needs)
5. **Click "Send"**

**WHY**: The service account needs permission to access your specific Google Sheet.
**HOW**: We share the sheet with the service account email address from the credentials file.

### Step 5: Configure the Application

1. **Open `config.py`** in your project directory
2. **Replace the placeholder values** with your actual data:

```python
# Google Sheets Configuration
SHEET_NAME = "Your Actual Sheet Name"  # Replace with your Google Sheet name
WORKSHEET_NAME = "Sheet1"  # Replace with your worksheet name
SHEET_ID = "your_actual_sheet_id_here"  # Replace with your Google Sheet ID

# Column Configuration - Replace with your actual column names
COLUMNS = {
    "date_column": "Date",  # Your date column name
    "value_column": "Sales",  # Your numerical value column name
    "category_column": "Product",  # Your category column name
    "name_column": "Name"  # Your name/label column name
}

# Chart Configuration
CHART_TITLE = "Sales Dashboard"  # Your desired chart title
X_AXIS_TITLE = "Products"  # Your X-axis label
Y_AXIS_TITLE = "Sales Amount"  # Your Y-axis label

# Refresh Configuration
REFRESH_INTERVAL = 30  # Refresh interval in seconds
```

**WHY**: The application needs to know which sheet and columns to use for visualization.
**HOW**: We update the configuration file with your specific sheet details and column names.

### Step 6: Find Your Google Sheet ID

The Google Sheet ID is found in your sheet's URL:
```
https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID_HERE/edit#gid=0
```

Copy the long string between `/d/` and `/edit` - that's your Sheet ID.

**WHY**: The Sheet ID uniquely identifies your specific Google Sheet.
**HOW**: We extract it from the URL to tell the API which sheet to access.

### Step 7: Run the Dashboard

Start the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

**WHY**: Streamlit provides an easy way to create interactive web applications.
**HOW**: We run the dashboard script which starts a local web server.

The dashboard will open in your browser at `http://localhost:8501`

## Usage

### Dashboard Features

1. **Data Overview**: Key metrics about your dataset
2. **Raw Data Table**: View and search through your data
3. **Interactive Charts**:
   - **Bar Chart**: Compare categories
   - **Line Chart**: Show trends over time
   - **Pie Chart**: Display proportions
4. **Auto-refresh**: Data updates automatically every 30 seconds
5. **Manual Refresh**: Click the refresh button to update immediately

### Customizing Charts

The dashboard automatically creates charts based on your column configuration:

- **Bar Chart**: Uses `category_column` (x-axis) and `value_column` (y-axis)
- **Line Chart**: Uses `date_column` (x-axis) and `value_column` (y-axis)
- **Pie Chart**: Uses `name_column` (labels) and `value_column` (values)

### Troubleshooting

#### Common Issues

1. **"Service account credentials file not found"**
   - Ensure `service_account_credentials.json` is in your project directory
   - Check the filename is exactly correct

2. **"Authentication failed"**
   - Verify your service account JSON file is valid
   - Ensure the Google Sheets API is enabled in your Google Cloud project

3. **"No data found in the specified sheet"**
   - Check your `SHEET_ID` in `config.py`
   - Verify the service account has access to the sheet
   - Ensure the worksheet name is correct

4. **"Columns not found in data"**
   - Update the column names in `config.py` to match your actual column names
   - Check for typos in column names

5. **Charts not displaying**
   - Ensure your data has the correct column types (numeric for values, dates for time series)
   - Check that your data doesn't have too many empty cells

#### Getting Help

If you encounter issues:

1. Check the error messages in the dashboard
2. Verify your Google Cloud project setup
3. Ensure your service account has the correct permissions
4. Double-check your configuration in `config.py`

## File Structure

```
project/
├── dashboard.py                          # Main Streamlit dashboard
├── google_sheets_connector.py            # Google Sheets API connector
├── config.py                             # Configuration file
├── requirements.txt                      # Python dependencies
├── service_account_credentials.json     # Google service account credentials (you add this)
└── README.md                            # This file
```

## Technical Details

### How It Works

1. **Authentication**: Uses Google service account for secure API access
2. **Data Fetching**: Connects to Google Sheets API and pulls data into Pandas DataFrame
3. **Caching**: Implements intelligent caching to avoid unnecessary API calls
4. **Visualization**: Creates interactive charts using Plotly
5. **Auto-refresh**: Uses Streamlit's session state and rerun mechanism for live updates

### Security

- Uses service account authentication (no user credentials stored)
- Read-only access to Google Sheets (configurable)
- Local data processing (no data sent to external services)

### Performance

- Intelligent caching reduces API calls
- Configurable refresh intervals
- Efficient data processing with Pandas
- Responsive UI with Streamlit

## Customization

### Adding New Chart Types

To add new chart types, create a new function in `dashboard.py` following the pattern of existing chart functions, then add a new tab in the main function.

### Modifying Refresh Behavior

Change the `REFRESH_INTERVAL` in `config.py` to adjust how often the data refreshes automatically.

### Styling

Modify the CSS in the `st.markdown()` section of `dashboard.py` to customize the appearance.

## License

This project is open source and available under the MIT License.

## Support

For questions or issues, please check the troubleshooting section above or create an issue in the project repository.
