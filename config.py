"""
Configuration file for Google Sheets to Streamlit Dashboard
Replace the placeholder values with your actual Google Sheets information
"""

# Google Sheets Configuration
SHEET_NAME = "scrapping data version 2"  # Your Google Sheet name
WORKSHEET_NAME = "Scraping Data version 2"  # Your worksheet name
SHEET_ID = "1aJ8HQ83HSE1xbzwIulTo-EbgS10fcz6LAitt-nZ7pOk"  # Your Google Sheet ID

# Multi-Sheet Configuration for Sales Funnel
SALES_FUNNEL_SHEETS = [
    {
        'name': 'Scraping Data version 2',
        'type': 'job_data',
        'description': 'Job applications and scraping data'
    },
    {
        'name': 'Proposals Tracking',
        'type': 'proposal_data', 
        'description': 'Proposals sent and tracking data'
    },
    {
        'name': 'Upwork Analytics',
        'type': 'analytics_data',
        'description': 'Upwork performance analytics data'
    },
    {
        'name': 'Upwork Ranking',
        'type': 'ranking_data',
        'description': 'Upwork ranking and positioning data'
    }
    # Add more sheets as needed for complete funnel analysis
    # {
    #     'name': 'Data',
    #     'type': 'general_data',
    #     'description': 'General data and metrics'
    # }
]

# Proposal Tracking Configuration
PROPOSAL_TRACKING_SHEET = "Proposals Tracking"  # Your proposal tracking sheet name
PROPOSAL_WORKSHEET = "Proposal Tracking"  # Your proposal tracking worksheet name

# Proposal Tracking Column Configuration
PROPOSAL_COLUMNS = {
    "applications_sent": "Applications Sent",  # Column B4
    "replies": "Replies",  # Column B5
    "interviews": "Interviews",  # Column B6
    "jobs_won": "Job Won",  # Column B7
    "average_deal_size": "Average Deal Size",  # Column B10
    "ltv_multiplier": "LTV Multiplier",  # Column B11
    "monthly_gross_revenue": "Monthly Gross Revenue",  # Column B12
    "target_applications": "Target Applications",  # Column E4
    "target_replies": "Target Replies",  # Column E5
    "target_interviews": "Target Interviews",  # Column E6
    "target_jobs_won": "Target Jobs Won",  # Column E7
    "target_deal_size": "Target Deal Size",  # Column E10
    "target_ltv": "Target LTV",  # Column E11
    "target_revenue": "Target Revenue",  # Column E12
    "impressions": "Impressions",  # Column H4
    "profile_views": "Profile Views",  # Column H5
    "invites": "Invites",  # Column H6
    "inbound_hires": "Inbound Hires",  # Column H7
    "inbound_deal_size": "Inbound Deal Size",  # Column H10
    "inbound_ltv": "Inbound LTV",  # Column H11
    "inbound_revenue": "Inbound Revenue",  # Column H12
    "daily_applications": "Daily Applications",  # Column M4
    "daily_replies": "Daily Replies",  # Column M5
    "daily_interviews": "Daily Interviews",  # Column M6
    "daily_jobs_won": "Daily Jobs Won",  # Column M7
    "daily_impressions": "Daily Impressions",  # Column M10
    "daily_profile_views": "Daily Profile Views",  # Column M11
    "weekly_invites": "Weekly Invites",  # Column M12
    "monthly_hires": "Monthly Hires"  # Column M13
}

# Column Configuration - Updated with actual column names from Excel formulas
COLUMNS = {
    "date_column": "Member since",  # Column name for dates/timestamps
    "publish_date_column": "Publish Date",  # Key column for time analysis (Column AD)
    "application_status_column": "Application status",  # Key column for status breakdown (Column AE)
    "value_column": "Score",  # Column name for numerical values
    "category_column": "Category",  # Column name for categories/groups
    "name_column": "Job Title",  # Column name for names/labels
    "icp_fit_column": "ICP Fit",  # Column I for ICP analysis
    "experience_level_column": "Experience Level",  # Column P for experience analysis
    "hourly_rate_column": "Hourly Rate",  # Column Q for rate analysis
    "country_column": "Country",  # Column T for location analysis
    "amount_spent_column": "Amount spent",  # Column W for budget analysis
    "proposals_column": "Proposals",  # Column Y for proposal tracking
    "interviewing_column": "Interviewing",  # Column Z for interview tracking
    "invite_sent_column": "Invite Sent",  # Column AA for invite tracking
    "unanswered_invites_column": "Unanswered Invites",  # Column AB for invite analysis
    "company_size_column": "Company size",  # Column AC for company analysis
    "comments_column": "Comments"  # Column AF for comments (if not interested)
}

# Chart Configuration
CHART_TITLE = "Job Scraping Data Dashboard"  # Your desired chart title
X_AXIS_TITLE = "Categories"  # Your X-axis label
Y_AXIS_TITLE = "Scores"  # Your Y-axis label

# Refresh Configuration
REFRESH_INTERVAL = 3600  # Refresh interval in seconds (1 hour = 3600 seconds)
