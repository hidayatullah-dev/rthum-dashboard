# 🚀 Quick Reference Card - Ultimate Dashboard

## 🏃‍♂️ Quick Start
```bash
# Test connection
python test_connection.py

# Run dashboard
streamlit run dashboard.py
```

## 📊 Top 10 Most Useful Formulas

### 1. ICP Score (Ideal Customer Profile)
```python
(Country == 'United States').astype(int) * 40 + 
(Amount spent > 20000).astype(int) * 30 + 
(Score > 30).astype(int) * 20 + 
(Proposals > 5).astype(int) * 10
```

### 2. Quality Score
```python
Score * 0.4 + (Amount spent / 1000) * 0.3 + Proposals * 0.3
```

### 3. High Value Jobs
```python
(Score > 40) & (Amount spent > 30000)
```

### 4. Quick Wins (Low Competition, Good Score)
```python
(Score > 30) & (Proposals < 10)
```

### 5. Response Rate
```python
(Interviewing / Proposals) * 100
```

### 6. Budget Efficiency
```python
(Score / Amount spent) * 1000
```

### 7. Competition Level
```python
Proposals / (Amount spent / 1000 + 1)
```

### 8. Python Jobs
```python
Job Title.str.contains('Python', case=False)
```

### 9. Remote Jobs
```python
Job Title.str.contains('Remote|Work from home', case=False)
```

### 10. Z-Score (Standardized)
```python
(Score - Score.mean()) / Score.std()
```

## 🧪 A/B Testing Examples

### Test 1: US vs International Jobs
- **Control**: `Country == 'United States'`
- **Treatment**: `Country != 'United States'`
- **Metric**: `Score`

### Test 2: High vs Low Budget
- **Control**: `Amount spent < 10000`
- **Treatment**: `Amount spent >= 10000`
- **Metric**: `Proposals`

### Test 3: Senior vs Junior Jobs
- **Control**: `~Job Title.str.contains('Senior|Lead|Principal', case=False)`
- **Treatment**: `Job Title.str.contains('Senior|Lead|Principal', case=False)`
- **Metric**: `Amount spent`

## 📈 Chart Types & When to Use

| Chart Type | Best For | Example Use |
|------------|----------|-------------|
| **Bar** | Categories comparison | Score by Category |
| **Line** | Trends over time | Score over time |
| **Scatter** | Correlation analysis | Score vs Amount spent |
| **Pie** | Part-to-whole | Jobs by Country |
| **Box** | Distribution analysis | Score distribution |
| **Heatmap** | Two-dimensional data | Country vs Category |

## ⚙️ Configuration Quick Fixes

### Google Sheets Connection
```python
# config.py
SHEET_ID = "your-sheet-id-here"
WORKSHEET_NAME = "your-worksheet-name"
```

### Column Names
```python
# Update these if your columns have different names
COLUMNS = {
    "date_column": "Member since",
    "value_column": "Score", 
    "category_column": "Category",
    "name_column": "Job Title"
}
```

## 🔧 Common Troubleshooting

### ❌ "Authentication failed"
- Check `service_account_credentials.json` exists
- Verify service account has sheet access
- Enable Google Sheets API

### ❌ "Failed to load data"
- Check `SHEET_ID` in config.py
- Verify `WORKSHEET_NAME` exists
- Test with `python test_connection.py`

### ❌ "Formula error"
- Check column names exist
- Use single expressions only
- Test with formula tester first

### ❌ "Chart not displaying"
- Check data has no NaN values
- Verify column data types
- Use safe chart function

## 🎯 Pro Tips

1. **Always test formulas** before creating them
2. **Use the formula library** for quick access to 50+ formulas
3. **Enable caching** for better performance
4. **Filter data** before analysis for faster processing
5. **Save experiments** to track your analysis over time

## 📱 Mobile Usage

The dashboard is fully responsive and works on:
- Desktop computers
- Tablets
- Mobile phones
- All modern browsers

## 🔄 Auto-Refresh

- **Default**: Every 1 hour (3600 seconds)
- **Manual**: Click "🔄 Refresh Data" button
- **Custom**: Change `REFRESH_INTERVAL` in config.py

## 📊 Key Metrics Explained

- **ICP Score**: 0-100, higher = better fit
- **Quality Score**: Weighted combination of score, budget, and proposals
- **Response Rate**: Percentage of proposals that get interviews
- **Competition Level**: Proposals per $1000 budget
- **Budget Efficiency**: Score per dollar spent

---

*Keep this card handy for quick reference while using the dashboard!*
