# 🚀 Ultimate Upwork Jobs Dashboard

A comprehensive data visualization dashboard designed to analyze Upwork job postings and help freelancers make data-driven decisions about which jobs to pursue.

## ✨ Features

- **📊 Real-time Data**: Connects to Google Sheets for live data updates
- **🧮 50+ Custom Formulas**: Advanced analytics for job analysis
- **📈 Interactive Visualizations**: Dynamic charts and graphs with Plotly
- **🧪 A/B Testing**: Built-in experiment framework
- **🔧 Custom Formula Builder**: Create your own analysis formulas
- **📱 Responsive UI**: Modern, mobile-friendly interface

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ultimate-upwork-dashboard.streamlit.app/)

## 📋 Quick Start

### Prerequisites
- Python 3.8+
- Google Sheets API access
- Service account credentials

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/hidayatullah-dev/ultimate-upwork-dashboard.git
cd ultimate-upwork-dashboard
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure Google Sheets**
   - Create a Google Cloud project
   - Enable Google Sheets API
   - Create a service account
   - Download the JSON credentials file
   - Rename it to `service_account_credentials.json`
   - Place it in the project root

4. **Update configuration**
   - Edit `config.py` with your Google Sheets details
   - Update `SHEET_ID`, `WORKSHEET_NAME`, and column names

5. **Test connection**
```bash
python test_connection.py
```

6. **Run the dashboard**
```bash
streamlit run dashboard.py
```

## 📚 Documentation

- **[Complete Guide Book](Ultimate_Dashboard_Guide_Book.md)** - Comprehensive documentation
- **[Quick Reference Card](Quick_Reference_Card.md)** - Quick access to formulas and tips
- **[Technical Documentation](Formula_Technical_Documentation.md)** - Deep dive into formulas
- **[Text Documentation](Ultimate_Dashboard_Documentation.txt)** - Plain text version

## 🧮 Formula Categories

### 1. Basic Scoring & Ranking
- Simple Score, Score Squared, Score Root
- Score Categories, Score Percentage

### 2. Financial & Budget Analysis
- Amount per Score, Budget Efficiency
- ROI Score, Value Score, Spend per Proposal

### 3. Geographic & Location Analysis
- US Jobs, International Jobs, Country Scoring
- Geographic Premium, Top Countries

### 4. Statistical & Mathematical
- Z-Score, Percentile Rank, Normalized Score
- Log Score, Exponential Score

### 5. Text & String Analysis
- Python Jobs, Senior Jobs, Remote Jobs
- Title Length, Word Count, Pattern Matching

### 6. Performance & Engagement
- Response Rate, Engagement Score
- Success Rate, Competition Level

### 7. Advanced Composite Scores
- ICP Score (Ideal Customer Profile)
- Quality Score, Priority Score, Risk Score
- Opportunity Score, Composite Index

## 🧪 A/B Testing Examples

### Test 1: US vs International Jobs
- **Control**: `Country == 'United States'`
- **Treatment**: `Country != 'United States'`
- **Metric**: `Score`

### Test 2: High vs Low Budget
- **Control**: `Amount spent < 10000`
- **Treatment**: `Amount spent >= 10000`
- **Metric**: `Proposals`

## 📊 Chart Types

| Chart Type | Best For | Example Use |
|------------|----------|-------------|
| **Bar** | Categories comparison | Score by Category |
| **Line** | Trends over time | Score over time |
| **Scatter** | Correlation analysis | Score vs Amount spent |
| **Pie** | Part-to-whole | Jobs by Country |
| **Box** | Distribution analysis | Score distribution |
| **Heatmap** | Two-dimensional data | Country vs Category |

## ⚙️ Configuration

### Google Sheets Setup
```python
# config.py
SHEET_ID = "your-sheet-id-here"
WORKSHEET_NAME = "your-worksheet-name"
SHEET_NAME = "your-sheet-name"
```

### Column Configuration
```python
COLUMNS = {
    "date_column": "Member since",
    "value_column": "Score",
    "category_column": "Category",
    "name_column": "Job Title"
}
```

## 🔧 Troubleshooting

### Common Issues

1. **Authentication failed**
   - Check `service_account_credentials.json` exists
   - Verify service account has sheet access
   - Enable Google Sheets API

2. **Failed to load data**
   - Check `SHEET_ID` in config.py
   - Verify `WORKSHEET_NAME` exists
   - Ensure service account has read access

3. **Formula error**
   - Check column names exist
   - Use single expressions only
   - Test with formula tester first

## 🚀 Streamlit Cloud Deployment

### Automatic Deployment
1. Fork this repository
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Connect your GitHub account
4. Select this repository
5. Deploy!

### Manual Deployment
1. Push your code to GitHub
2. Add your `service_account_credentials.json` as a secret in Streamlit Cloud
3. Set environment variables if needed
4. Deploy

## 📁 Project Structure

```
ultimate-upwork-dashboard/
├── dashboard.py                    # Main dashboard application
├── ultimate_dashboard.py          # Enhanced dashboard version
├── google_sheets_connector.py     # Google Sheets API integration
├── config.py                      # Configuration settings
├── test_connection.py             # Connection testing utility
├── requirements.txt               # Python dependencies
├── service_account_credentials.json # Google API credentials (not in repo)
├── README.md                      # This file
├── Ultimate_Dashboard_Guide_Book.md # Complete guide
├── Quick_Reference_Card.md        # Quick reference
├── Formula_Technical_Documentation.md # Technical docs
└── Ultimate_Dashboard_Documentation.txt # Text documentation
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [Plotly](https://plotly.com/) for interactive visualizations
- [Google Sheets API](https://developers.google.com/sheets) for data integration
- [Pandas](https://pandas.pydata.org/) for data manipulation

## 📞 Support

If you have any questions or need help:

1. Check the [documentation](Ultimate_Dashboard_Guide_Book.md)
2. Look at the [troubleshooting section](#troubleshooting)
3. Open an [issue](https://github.com/hidayatullah-dev/ultimate-upwork-dashboard/issues)
4. Contact the maintainer

---

**Made with ❤️ for the freelancer community**