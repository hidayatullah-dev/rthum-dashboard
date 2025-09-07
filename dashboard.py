"""
Ultimate Upwork Jobs Dashboard
Comprehensive dashboard with advanced features, custom formulas, experiments, and beautiful UI
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from google_sheets_connector import GoogleSheetsConnector
from config import *
import os
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Ultimate Upwork Jobs Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS for modern UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    if 'data_cache' not in st.session_state:
        st.session_state.data_cache = None
    
    if 'cache_timestamp' not in st.session_state:
        st.session_state.cache_timestamp = None
    
    if 'experiments' not in st.session_state:
        st.session_state.experiments = {}
    
    if 'custom_formulas' not in st.session_state:
        st.session_state.custom_formulas = {}

def should_refresh_data():
    """Determine if the data should be refreshed based on time interval."""
    current_time = datetime.now()
    time_since_refresh = current_time - st.session_state.last_refresh
    return time_since_refresh.total_seconds() >= REFRESH_INTERVAL

def load_data():
    """Load data from Google Sheets with caching and error handling."""
    try:
        if (st.session_state.data_cache is not None and 
            st.session_state.cache_timestamp is not None and 
            not should_refresh_data()):
            return st.session_state.data_cache
        
        # Try to use Streamlit secrets first (for cloud deployment)
        if 'gcp_service_account' in st.secrets:
            # Create credentials from Streamlit secrets
            import json
            from google.oauth2.service_account import Credentials
            
            # Convert secrets to credentials format
            credentials_info = dict(st.secrets['gcp_service_account'])
            credentials = Credentials.from_service_account_info(credentials_info)
            
            # Create connector with credentials
            connector = GoogleSheetsConnector.from_credentials(credentials)
            df = connector.get_sheet_data(SHEET_ID, WORKSHEET_NAME)
        else:
            # Fallback to local file (for local development)
            credentials_path = "service_account_credentials.json"
            
            if not os.path.exists(credentials_path):
                st.error(f"❌ Service account credentials not found. Please add them to Streamlit secrets or place {credentials_path} in the project directory.")
                return None
            
            connector = GoogleSheetsConnector(credentials_path)
            df = connector.get_sheet_data(SHEET_ID, WORKSHEET_NAME)
        
        if df is not None:
            df = clean_and_enhance_data(df)
            st.session_state.data_cache = df
            st.session_state.cache_timestamp = datetime.now()
            st.session_state.last_refresh = datetime.now()
            return df
        else:
            st.error("❌ Failed to load data from Google Sheets")
            return None
            
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

def clean_and_enhance_data(df):
    """Clean and enhance data with additional calculated fields."""
    # Convert numeric columns
    numeric_columns = ['Score', 'Amount spent', 'Active hires', 'Proposals', 'Interviewing', 'Invite Sent', 'Unanswered Invites']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert date columns
    if 'Member since' in df.columns:
        df['Member since'] = pd.to_datetime(df['Member since'], errors='coerce')
    
    # Clean text columns
    text_columns = ['Job Title', 'Category', 'Country', 'City', 'Industry', 'Action', 'ICP Fit']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    # Add calculated fields
    df = add_calculated_fields(df)
    
    return df

def add_calculated_fields(df):
    """Add calculated fields for better analysis."""
    # Score categories
    if 'Score' in df.columns:
        df['Score_Category'] = pd.cut(df['Score'], 
                                    bins=[0, 20, 40, 60, 100], 
                                    labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Amount spent categories
    if 'Amount spent' in df.columns:
        df['Spend_Category'] = pd.cut(df['Amount spent'], 
                                    bins=[0, 1000, 10000, 50000, float('inf')], 
                                    labels=['Low', 'Medium', 'High', 'Very High'])
    
    # ICP Score (custom formula)
    if all(col in df.columns for col in ['Country', 'Amount spent', 'Score']):
        df['ICP_Score'] = calculate_icp_score(df)
    
    # Response Rate
    if all(col in df.columns for col in ['Proposals', 'Interviewing']):
        df['Response_Rate'] = (df['Interviewing'] / df['Proposals'] * 100).fillna(0)
    
    # Job Quality Score
    if all(col in df.columns for col in ['Score', 'Amount spent', 'Proposals']):
        df['Quality_Score'] = (df['Score'] * 0.4 + 
                              (df['Amount spent'] / 1000) * 0.3 + 
                              df['Proposals'] * 0.3).fillna(0)
    
    return df

def calculate_icp_score(df):
    """Calculate Ideal Customer Profile score."""
    score = 0
    
    # Geography (USA/UAE = 40 points)
    if 'Country' in df.columns:
        usa_uae = df['Country'].isin(['United States', 'UAE', 'United Arab Emirates', 'USA'])
        score += usa_uae.astype(int) * 40
    
    # Budget (Amount spent > 20k = 30 points)
    if 'Amount spent' in df.columns:
        high_spend = df['Amount spent'] > 20000
        score += high_spend.astype(int) * 30
    
    # Score (High score = 20 points)
    if 'Score' in df.columns:
        high_score = df['Score'] > 30
        score += high_score.astype(int) * 20
    
    # Proposals (High proposals = 10 points)
    if 'Proposals' in df.columns:
        high_proposals = df['Proposals'] > 5
        score += high_proposals.astype(int) * 10
    
    return score

def create_advanced_metrics(df):
    """Create advanced KPI metrics with better styling."""
    if df is None or df.empty:
        return
    
    # Calculate metrics
    total_jobs = len(df)
    avg_score = df['Score'].mean() if 'Score' in df.columns else 0
    total_amount = df['Amount spent'].sum() if 'Amount spent' in df.columns else 0
    avg_proposals = df['Proposals'].mean() if 'Proposals' in df.columns else 0
    icp_jobs = len(df[df['ICP_Score'] > 70]) if 'ICP_Score' in df.columns else 0
    high_quality = len(df[df['Quality_Score'] > 50]) if 'Quality_Score' in df.columns else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <h3>📊 Total Jobs</h3>
            <h1 style="font-size: 3rem; margin: 0;">{total_jobs:,}</h1>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">Jobs Analyzed</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <h3>⭐ Average Score</h3>
            <h1 style="font-size: 3rem; margin: 0;">{avg_score:.1f}</h1>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">Quality Rating</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <h3>💰 Total Value</h3>
            <h1 style="font-size: 3rem; margin: 0;">${total_amount:,.0f}</h1>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">Amount Spent</p>
        </div>
        ''', unsafe_allow_html=True)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown(f'''
        <div class="metric-card">
            <h3>🎯 ICP Jobs</h3>
            <h1 style="font-size: 3rem; margin: 0;">{icp_jobs}</h1>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">Ideal Customer Profile</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col5:
        st.markdown(f'''
        <div class="metric-card">
            <h3>📝 Avg Proposals</h3>
            <h1 style="font-size: 3rem; margin: 0;">{avg_proposals:.1f}</h1>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">Per Job</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col6:
        st.markdown(f'''
        <div class="metric-card">
            <h3>🏆 High Quality</h3>
            <h1 style="font-size: 3rem; margin: 0;">{high_quality}</h1>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">Quality Jobs</p>
        </div>
        ''', unsafe_allow_html=True)

def create_safe_chart(df, chart_type, x_col, y_col, color_col=None, size_col=None, title="Custom Chart"):
    """Create a safe chart that handles NaN values properly."""
    try:
        # Clean data - remove rows with NaN values in required columns
        df_clean = df.dropna(subset=[x_col, y_col])
        
        if df_clean.empty:
            st.warning("⚠️ No valid data to plot after cleaning")
            return None
        
        # Handle size column for scatter plots
        if chart_type == 'scatter' and size_col and size_col in df_clean.columns:
            # Clean size column and convert to numeric
            df_clean[size_col] = pd.to_numeric(df_clean[size_col], errors='coerce')
            df_clean = df_clean.dropna(subset=[size_col])
            
            # Ensure size values are positive and not too large
            df_clean[size_col] = df_clean[size_col].clip(lower=1, upper=100)
        
        if chart_type == 'bar':
            fig = px.bar(df_clean, x=x_col, y=y_col, color=color_col, title=title,
                        color_discrete_sequence=px.colors.qualitative.Set3)
        elif chart_type == 'line':
            fig = px.line(df_clean, x=x_col, y=y_col, color=color_col, title=title,
                         markers=True, line_shape='smooth')
        elif chart_type == 'scatter':
            if size_col and size_col in df_clean.columns:
                fig = px.scatter(df_clean, x=x_col, y=y_col, color=color_col, size=size_col, title=title)
            else:
                fig = px.scatter(df_clean, x=x_col, y=y_col, color=color_col, title=title)
        elif chart_type == 'pie':
            fig = px.pie(df_clean, names=x_col, values=y_col, title=title,
                        color_discrete_sequence=px.colors.qualitative.Pastel)
        elif chart_type == 'heatmap':
            if color_col and color_col in df_clean.columns:
                pivot_data = df_clean.pivot_table(values=y_col, index=x_col, 
                                                columns=color_col, aggfunc='mean', fill_value=0)
                fig = px.imshow(pivot_data, title=title, color_continuous_scale='Blues')
            else:
                st.warning("Heatmap requires a color column")
                return None
        elif chart_type == 'box':
            fig = px.box(df_clean, x=x_col, y=y_col, color=color_col, title=title)
        elif chart_type == 'violin':
            fig = px.violin(df_clean, x=x_col, y=y_col, color=color_col, title=title)
        elif chart_type == 'histogram':
            fig = px.histogram(df_clean, x=x_col, color=color_col, title=title)
        else:
            fig = px.bar(df_clean, x=x_col, y=y_col, color=color_col, title=title)
        
        # Enhanced styling
        fig.update_layout(
            title_font_size=20,
            title_font_color='#2c3e50',
            font=dict(family='Inter, sans-serif', size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Inter"
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def create_experiment_interface(df):
    """Create A/B testing and experiment interface."""
    st.subheader("🧪 Experiments & A/B Testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Create New Experiment")
        
        experiment_name = st.text_input("Experiment Name:", key="exp_name")
        
        # Control Group Filter
        st.markdown("**Control Group Filter:**")
        control_filter = st.text_area("Filter (e.g., Score > 30 AND Country == 'United States'):", 
                                    key="control_filter")
        
        # Treatment Group Filter
        st.markdown("**Treatment Group Filter:**")
        treatment_filter = st.text_area("Filter (e.g., Score > 40 AND Amount spent > 20000):", 
                                      key="treatment_filter")
        
        # Metric to Compare
        metric = st.selectbox("Metric to Compare:", 
                            ['Score', 'Amount spent', 'Proposals', 'Quality_Score', 'ICP_Score'],
                            key="exp_metric")
        
        if st.button("🚀 Run Experiment", key="run_exp"):
            if experiment_name and control_filter and treatment_filter:
                run_experiment(df, experiment_name, control_filter, treatment_filter, metric)
            else:
                st.warning("Please fill in all fields")
    
    with col2:
        st.markdown("### Experiment Results")
        
        if st.session_state.experiments:
            for exp_name, results in st.session_state.experiments.items():
                with st.expander(f"📊 {exp_name}"):
                    st.json(results)

def run_experiment(df, name, control_filter, treatment_filter, metric):
    """Run A/B test experiment."""
    try:
        # Apply filters
        control_group = df.query(control_filter) if control_filter else df
        treatment_group = df.query(treatment_filter) if treatment_filter else df
        
        if control_group.empty or treatment_group.empty:
            st.error("One or both groups are empty. Check your filters.")
            return
        
        # Calculate metrics
        control_mean = control_group[metric].mean()
        treatment_mean = treatment_group[metric].mean()
        
        control_std = control_group[metric].std()
        treatment_std = treatment_group[metric].std()
        
        # Calculate statistical significance (simplified)
        n_control = len(control_group)
        n_treatment = len(treatment_group)
        
        # Store results
        results = {
            'control_group': {
                'size': n_control,
                'mean': float(control_mean),
                'std': float(control_std)
            },
            'treatment_group': {
                'size': n_treatment,
                'mean': float(treatment_mean),
                'std': float(treatment_std)
            },
            'difference': float(treatment_mean - control_mean),
            'percent_change': float((treatment_mean - control_mean) / control_mean * 100) if control_mean != 0 else 0,
            'metric': metric,
            'timestamp': datetime.now().isoformat()
        }
        
        st.session_state.experiments[name] = results
        
        # Display results
        st.success(f"✅ Experiment '{name}' completed!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Control Mean", f"{control_mean:.2f}")
        with col2:
            st.metric("Treatment Mean", f"{treatment_mean:.2f}")
        with col3:
            st.metric("Difference", f"{results['difference']:.2f}")
        
    except Exception as e:
        st.error(f"Error running experiment: {str(e)}")

def create_formula_builder(df):
    """Create custom formula builder interface."""
    st.subheader("🧮 Custom Formula Builder")
    
    # Show available columns
    st.markdown("### 📋 Available Columns")
    col_columns = st.columns(4)
    for i, col in enumerate(df.columns.tolist()):
        with col_columns[i % 4]:
            st.code(col, language="text")
    
    # Quick formula tester
    st.markdown("### 🧪 Quick Formula Tester")
    with st.expander("Test formulas before creating them"):
        test_formula = st.text_input("Test Formula:", placeholder="Score > 30", key="test_formula")
        if st.button("🔍 Test Formula", key="test_formula_btn"):
            if test_formula:
                try:
                    # Handle column names with spaces
                    df_temp = df.copy()
                    column_mapping = {}
                    
                    # Create safe column names (replace spaces with underscores)
                    for col in df_temp.columns:
                        safe_col = col.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
                        if safe_col != col:
                            column_mapping[col] = safe_col
                            df_temp[safe_col] = df_temp[col]
                    
                    # Replace column names in expression with safe names
                    test_formula_safe = test_formula
                    for original, safe in column_mapping.items():
                        test_formula_safe = test_formula_safe.replace(original, safe)
                    
                    test_result = df_temp.eval(test_formula_safe, engine='python')
                    st.success("✅ Formula is valid!")
                    st.write("**Sample results:**")
                    st.write(test_result.head(10))
                    st.write(f"**Data type:** {test_result.dtype}")
                    st.write(f"**Unique values:** {test_result.nunique()}")
                except Exception as e:
                    st.error(f"❌ Formula error: {str(e)}")
            else:
                st.warning("Please enter a formula to test")
    
    # Test formulas library
    st.markdown("### 📚 Test Formulas Library (50+ Complex Formulas)")
    with st.expander("Click to see 50+ ready-to-use complex formulas"):
        create_test_formulas_library(df)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Create New Formula")
        
        formula_name = st.text_input("Formula Name:", key="formula_name", 
                                   placeholder="e.g., High Value Score")
        
        # Show example formulas
        with st.expander("💡 Example Formulas"):
            st.code("""
# Mathematical operations:
Score * 2
(Amount spent / 1000) + Score
Score * 0.4 + (Amount spent / 1000) * 0.3

# Boolean expressions:
Score > 30
Country == 'United States'
(Score > 30) & (Amount spent > 20000)

# String operations:
Job Title.str.contains('Python')
Category.str.upper()

# Statistical functions:
Score.mean()
Amount spent.max()
            """, language="python")
        
        formula_expression = st.text_area(
            "Formula Expression:",
            placeholder="Example: (Score * 0.4) + (Amount spent / 1000 * 0.3) + (Proposals * 0.2)",
            key="formula_expr",
            height=100
        )
        
        # Validation tips
        st.info("💡 **Formula Rules:**\n- Use single expressions only\n- No semicolons or newlines\n- Use column names as variables\n- Supported: +, -, *, /, **, &, |, ==, !=, >, <, >=, <=")
        
        if st.button("🔧 Create Formula", key="create_formula"):
            if formula_name and formula_expression:
                create_custom_formula(df, formula_name, formula_expression)
            else:
                st.warning("Please fill in both formula name and expression")
    
    with col2:
        st.markdown("### Saved Formulas")
        
        if st.session_state.custom_formulas:
            for formula_name, formula_data in st.session_state.custom_formulas.items():
                with st.expander(f"📐 {formula_name}"):
                    st.code(formula_data['expression'], language="python")
                    st.caption(f"Created: {formula_data['timestamp']}")
                    
                    col_apply, col_delete = st.columns(2)
                    with col_apply:
                        if st.button(f"Apply", key=f"apply_{formula_name}"):
                            apply_formula(df, formula_name, formula_data['expression'])
                    with col_delete:
                        if st.button(f"Delete", key=f"delete_{formula_name}"):
                            del st.session_state.custom_formulas[formula_name]
                            st.rerun()
        else:
            st.info("No formulas created yet. Create your first formula on the left!")

def create_custom_formula(df, name, expression):
    """Create and apply custom formula."""
    try:
        # Clean and validate the expression
        expression = expression.strip()
        
        # Check if expression is empty
        if not expression:
            st.error("❌ Please enter a formula expression.")
            return
        
        # Check for obvious multi-statement indicators
        if ';' in expression:
            st.error("❌ Semicolons are not allowed. Use single expressions only.")
            return
        
        # Check for multiple lines (but allow single line breaks in strings)
        lines = expression.split('\n')
        if len(lines) > 1:
            # Check if it's just whitespace on additional lines
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            if len(non_empty_lines) > 1:
                st.error("❌ Multiple formulas detected. Please enter only ONE formula at a time.")
                st.info("💡 **You can only create one formula at a time. Try these one by one:**")
                st.code("""
Score > 30
Country == 'United States'
(Score > 30) & (Amount spent > 20000)
(Country == 'United States') | (Country == 'UAE')
(Score >= 20) & (Score <= 80)
Country != 'Unknown'
                """)
                return
        
        # Handle column names with spaces by creating a temporary dataframe with safe column names
        df_temp = df.copy()
        column_mapping = {}
        reverse_mapping = {}
        
        # Create safe column names (replace spaces with underscores)
        for col in df_temp.columns:
            safe_col = col.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
            if safe_col != col:
                column_mapping[col] = safe_col
                reverse_mapping[safe_col] = col
                df_temp[safe_col] = df_temp[col]
        
        # Replace column names in expression with safe names
        expression_safe = expression
        for original, safe in column_mapping.items():
            expression_safe = expression_safe.replace(original, safe)
        
        # Validate expression by trying to evaluate it
        test_result = df_temp.eval(expression_safe, engine='python')
        
        # Store formula
        st.session_state.custom_formulas[name] = {
            'expression': expression,
            'timestamp': datetime.now().isoformat()
        }
        
        # Apply formula
        df[f'Custom_{name}'] = test_result
        
        st.success(f"✅ Formula '{name}' created and applied!")
        st.info(f"New column 'Custom_{name}' added to dataset")
        
        # Show preview of the new column
        if len(df) > 0:
            st.write("**Preview of new column:**")
            st.write(df[f'Custom_{name}'].head(10))
            
            # Add option to create chart from formula
            st.markdown("---")
            st.markdown("### 📊 Create Chart from Formula")
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                chart_type = st.selectbox(
                    "Chart Type:",
                    ["bar", "line", "scatter", "pie", "histogram", "box"],
                    key=f"chart_type_{name}"
                )
            
            with col_chart2:
                if chart_type in ['scatter', 'line']:
                    x_column = st.selectbox(
                        "X-Axis:",
                        df.columns.tolist(),
                        key=f"x_column_{name}"
                    )
                else:
                    x_column = f'Custom_{name}'
            
            if st.button(f"🎨 Create Chart", key=f"create_chart_{name}"):
                create_formula_chart(df, f'Custom_{name}', chart_type, x_column, name)
        
    except Exception as e:
        error_msg = str(e)
        
        # Provide specific error messages for common issues
        if "only a single expression is allowed" in error_msg:
            st.error("❌ Only single expressions are allowed. Check for semicolons or multiple statements.")
            st.info("💡 **Valid examples:**")
            st.code("Score * 2\n(Amount spent / 1000) + Score\nScore > 30")
        elif "name" in error_msg.lower() and "is not defined" in error_msg.lower():
            st.error("❌ Column name not found. Check the available columns list.")
            st.info("💡 **Available columns:**")
            st.write(", ".join(df.columns.tolist()[:15]) + ("..." if len(df.columns) > 15 else ""))
        else:
            st.error(f"Error creating formula: {error_msg}")
        
        st.info("💡 **Formula Tips:**")
        st.write("- Use exact column names from the list above")
        st.write("- Use single expressions only (no semicolons)")
        st.write("- Supported operators: +, -, *, /, **, &, |, ==, !=, >, <, >=, <=")
        st.write("- Use parentheses for complex expressions")

def apply_formula(df, name, expression):
    """Apply existing formula to dataset."""
    try:
        # Clean and validate the expression
        expression = expression.strip()
        
        # Check if expression is empty
        if not expression:
            st.error("❌ Formula expression is empty.")
            return
        
        # Check for obvious multi-statement indicators
        if ';' in expression:
            st.error("❌ Semicolons are not allowed. Use single expressions only.")
            return
        
        # Check for multiple lines
        lines = expression.split('\n')
        if len(lines) > 1:
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            if len(non_empty_lines) > 1:
                st.error("❌ Multiple lines are not allowed. Use single expressions only.")
                return
        
        # Handle column names with spaces by creating a temporary dataframe with safe column names
        df_temp = df.copy()
        column_mapping = {}
        reverse_mapping = {}
        
        # Create safe column names (replace spaces with underscores)
        for col in df_temp.columns:
            safe_col = col.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
            if safe_col != col:
                column_mapping[col] = safe_col
                reverse_mapping[safe_col] = col
                df_temp[safe_col] = df_temp[col]
        
        # Replace column names in expression with safe names
        expression_safe = expression
        for original, safe in column_mapping.items():
            expression_safe = expression_safe.replace(original, safe)
        
        # Apply formula
        df[f'Custom_{name}'] = df_temp.eval(expression_safe, engine='python')
        st.success(f"✅ Formula '{name}' applied!")
        
        # Show preview
        if len(df) > 0:
            st.write("**Preview of updated column:**")
            st.write(df[f'Custom_{name}'].head(10))
            
    except Exception as e:
        error_msg = str(e)
        
        if "only a single expression is allowed" in error_msg:
            st.error("❌ Only single expressions are allowed. Check for semicolons or multiple statements.")
        elif "name" in error_msg.lower() and "is not defined" in error_msg.lower():
            st.error("❌ Column name not found. Check the available columns list.")
            st.info("💡 **Available columns:**")
            st.write(", ".join(df.columns.tolist()[:15]) + ("..." if len(df.columns) > 15 else ""))
        else:
            st.error(f"Error applying formula: {error_msg}")
        
        st.info("💡 Check that all column names in the formula exist in your data.")

def create_test_formulas_library(df):
    """Create a comprehensive library of 50+ test formulas with different categories."""
    
    # Formula categories
    categories = {
        "🎯 Basic Scoring & Ranking": [
            ("Simple Score", "Score", "Basic job score"),
            ("Score Squared", "Score ** 2", "Exponential scoring"),
            ("Score Root", "Score ** 0.5", "Square root scoring"),
            ("Score Percentage", "(Score / 100) * 100", "Score as percentage"),
            ("Score Categories", "((Score >= 0) & (Score < 20)) * 1 + ((Score >= 20) & (Score < 40)) * 2 + ((Score >= 40) & (Score < 60)) * 3 + (Score >= 60) * 4", "Categorical scoring (1=Low, 2=Medium, 3=High, 4=Very High)"),
        ],
        
        "💰 Financial & Budget Analysis": [
            ("Amount per Score", "Amount spent / Score", "Budget efficiency"),
            ("High Budget Jobs", "Amount spent > 50000", "Premium jobs filter"),
            ("Budget Categories", "((Amount spent >= 0) & (Amount spent < 1000)) * 1 + ((Amount spent >= 1000) & (Amount spent < 10000)) * 2 + ((Amount spent >= 10000) & (Amount spent < 50000)) * 3 + (Amount spent >= 50000) * 4", "Budget classification (1=Low, 2=Medium, 3=High, 4=Very High)"),
            ("Value Score", "Score * (Amount spent / 1000)", "Value-based scoring"),
            ("Budget Efficiency", "(Score / Amount spent) * 1000", "Efficiency metric"),
            ("Spend per Proposal", "Amount spent / Proposals", "Cost per proposal"),
            ("ROI Score", "(Score * Amount spent) / 100000", "Return on investment"),
        ],
        
        "🌍 Geographic & Location Analysis": [
            ("US Jobs", "Country == 'United States'", "US-based jobs"),
            ("International Jobs", "Country != 'United States'", "Non-US jobs"),
            ("Top Countries", "Country.isin(['United States', 'UAE', 'Canada', 'UK'])", "Major markets"),
            ("Country Score", "Country.map({'United States': 100, 'UAE': 90, 'Canada': 80, 'UK': 70}).fillna(50)", "Country scoring"),
            ("Geographic Premium", "(Country == 'United States').astype(int) * 20", "US premium"),
        ],
        
        "📊 Statistical & Mathematical": [
            ("Z-Score", "(Score - Score.mean()) / Score.std()", "Standardized score"),
            ("Percentile Rank", "Score.rank(pct=True) * 100", "Percentile ranking"),
            ("Log Score", "Score.apply(lambda x: __import__('math').log(x + 1) if x > 0 else 0)", "Logarithmic transformation"),
            ("Exponential Score", "Score.apply(lambda x: __import__('math').exp(x / 20))", "Exponential transformation"),
            ("Normalized Score", "(Score - Score.min()) / (Score.max() - Score.min())", "Min-max normalization"),
            ("Score Mean", "Score.mean()", "Average score"),
        ],
        
        "🔍 Text & String Analysis": [
            ("Python Jobs", "Job Title.str.contains('Python', case=False)", "Python-related jobs"),
            ("Senior Jobs", "Job Title.str.contains('Senior|Lead|Principal', case=False)", "Senior positions"),
            ("Remote Jobs", "Job Title.str.contains('Remote|Work from home', case=False)", "Remote work"),
            ("Title Length", "Job Title.str.len()", "Job title character count"),
            ("Word Count", "Job Title.str.split().str.len()", "Words in job title"),
            ("Has Numbers", "Job Title.str.contains(r'\\d+')", "Titles with numbers"),
            ("All Caps", "Job Title.str.isupper()", "All uppercase titles"),
        ],
        
        "📈 Performance & Engagement": [
            ("Response Rate", "(Interviewing / Proposals) * 100", "Interview response rate"),
            ("Engagement Score", "Proposals + Interviewing + Invite Sent", "Total engagement"),
            ("Activity Level", "Active hires + Proposals + Interviewing", "Activity metric"),
            ("Success Rate", "(Interviewing / (Proposals + 1)) * 100", "Success percentage"),
            ("Competition Level", "Proposals / (Amount spent / 1000 + 1)", "Competition metric"),
            ("Urgency Score", "Proposals * 10 + Interviewing * 5", "Urgency indicator"),
        ],
        
        "🎨 Advanced Composite Scores": [
            ("ICP Score", "(Country == 'United States').astype(int) * 40 + (Amount spent > 20000).astype(int) * 30 + (Score > 30).astype(int) * 20 + (Proposals > 5).astype(int) * 10", "Ideal Customer Profile"),
            ("Quality Score", "Score * 0.4 + (Amount spent / 1000) * 0.3 + Proposals * 0.3", "Quality assessment"),
            ("Priority Score", "Score * 0.3 + (Amount spent / 1000) * 0.4 + Proposals * 0.3", "Priority ranking"),
            ("Risk Score", "100 - (Proposals * 10)", "Risk assessment"),
            ("Opportunity Score", "Score * (Amount spent / 1000) * Proposals", "Opportunity value"),
            ("Composite Index", "(Score * 0.25) + ((Amount spent / 1000) * 0.25) + (Proposals * 0.25) + ((Interviewing / Proposals) * 100 * 0.25)", "Multi-factor index"),
        ],
        
        "⏰ Time-Based Analysis": [
            ("Recent Jobs", "Member since > '2024-01-01'", "Recent member jobs"),
            ("Veteran Jobs", "Member since < '2020-01-01'", "Established member jobs"),
            ("Member Duration", "Member since.apply(lambda x: (__import__('datetime').datetime.now() - x).days if pd.notna(x) else 0)", "Days since joining"),
            ("New Member", "Member since.apply(lambda x: (__import__('datetime').datetime.now() - x).days <= 30 if pd.notna(x) else False)", "New members"),
            ("Join Year", "Member since.dt.year", "Year joined"),
        ],
        
        "🔢 Boolean Logic & Conditions": [
            ("High Value", "(Score > 40) & (Amount spent > 30000)", "High value jobs"),
            ("Quick Wins", "(Score > 30) & (Proposals < 10)", "Low competition, good score"),
            ("Premium Market", "(Country == 'United States') & (Amount spent > 50000)", "US premium jobs"),
            ("Active Clients", "(Active hires > 0) & (Amount spent > 10000)", "Active spenders"),
            ("High Competition", "Proposals > 50", "Highly competitive jobs"),
            ("Easy Targets", "(Score > 25) & (Proposals < 5) & (Amount spent > 5000)", "Easy targets"),
        ],
        
        "📊 Advanced Analytics": [
            ("Score Range", "Score.max() - Score.min()", "Score range"),
            ("Amount Median", "Amount spent.median()", "Median budget"),
            ("Proposal Density", "Proposals / (Amount spent / 1000 + 1)", "Proposal density"),
            ("Efficiency Ratio", "Score / (Proposals + 1)", "Efficiency metric"),
            ("Score Rank", "Score.rank()", "Score ranking"),
            ("Growth Potential", "(Amount spent / Score) * Proposals", "Growth indicator"),
        ]
    }
    
    # Create tabs for each category
    tab_names = list(categories.keys())
    tabs = st.tabs(tab_names)
    
    for i, (category, formulas) in enumerate(categories.items()):
        with tabs[i]:
            st.markdown(f"### {category}")
            
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            for j, (name, formula, description) in enumerate(formulas):
                with col1 if j % 2 == 0 else col2:
                    st.markdown(f"**{name}**")
                    st.code(formula, language="python")
                    st.caption(description)
                    
                    # Add buttons for each formula
                    if st.button(f"Test {name}", key=f"test_{name}_{j}"):
                        test_single_formula(df, formula, name)
                    if st.button(f"Create {name}", key=f"create_{name}_{j}"):
                        create_formula_from_library(df, name, formula)
                    
                    st.markdown("---")

def test_single_formula(df, formula, name):
    """Test a single formula from the library."""
    try:
        # Handle column names with spaces
        df_temp = df.copy()
        column_mapping = {}
        
        # Create safe column names
        for col in df_temp.columns:
            safe_col = col.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
            if safe_col != col:
                column_mapping[col] = safe_col
                df_temp[safe_col] = df_temp[col]
        
        # Replace column names in expression
        formula_safe = formula
        for original, safe in column_mapping.items():
            formula_safe = formula_safe.replace(original, safe)
        
        # Evaluate formula
        result = df_temp.eval(formula_safe, engine='python')
        
        st.success(f"✅ {name} formula is valid!")
        
        # Show results in a simpler format
        st.write(f"**Sample Values:** {len(result)} rows")
        st.write(f"**Data Type:** {str(result.dtype)}")
        st.write(f"**Unique Values:** {result.nunique()}")
        
        # Show sample data
        st.write("**Sample Results:**")
        st.write(result.head(10))
        
        # Show statistics for numerical data
        if pd.api.types.is_numeric_dtype(result):
            st.write("**Statistics:**")
            st.write(result.describe())
        
    except Exception as e:
        st.error(f"❌ Error testing {name}: {str(e)}")

def create_formula_from_library(df, name, formula):
    """Create a formula directly from the library."""
    try:
        # Handle column names with spaces
        df_temp = df.copy()
        column_mapping = {}
        
        # Create safe column names
        for col in df_temp.columns:
            safe_col = col.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
            if safe_col != col:
                column_mapping[col] = safe_col
                df_temp[safe_col] = df_temp[col]
        
        # Replace column names in expression
        formula_safe = formula
        for original, safe in column_mapping.items():
            formula_safe = formula_safe.replace(original, safe)
        
        # Evaluate and store
        result = df_temp.eval(formula_safe, engine='python')
        df[f'Custom_{name}'] = result
        
        # Store in session state
        st.session_state.custom_formulas[name] = {
            'expression': formula,
            'timestamp': datetime.now().isoformat()
        }
        
        st.success(f"✅ Formula '{name}' created and applied!")
        st.info(f"New column 'Custom_{name}' added to dataset")
        
        # Show preview
        st.write("**Preview of new column:**")
        st.write(df[f'Custom_{name}'].head(10))
        
    except Exception as e:
        st.error(f"Error creating formula {name}: {str(e)}")

def create_formula_chart(df, formula_column, chart_type, x_column, formula_name):
    """Create a chart from a formula result."""
    try:
        if chart_type == 'bar':
            if x_column == formula_column:
                # Count values for categorical data
                value_counts = df[formula_column].value_counts()
                fig = px.bar(x=value_counts.index, y=value_counts.values, 
                           title=f"Distribution of {formula_name}")
            else:
                fig = px.bar(df, x=x_column, y=formula_column, 
                           title=f"{formula_name} by {x_column}")
        
        elif chart_type == 'line':
            fig = px.line(df, x=x_column, y=formula_column, 
                         title=f"{formula_name} over {x_column}")
        
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=x_column, y=formula_column, 
                           title=f"{formula_name} vs {x_column}")
        
        elif chart_type == 'pie':
            if df[formula_column].dtype in ['bool', 'object']:
                # For categorical/boolean data, show distribution
                value_counts = df[formula_column].value_counts()
                fig = px.pie(values=value_counts.values, names=value_counts.index, 
                           title=f"Distribution of {formula_name}")
            else:
                # For numerical data, create bins
                df_temp = df.copy()
                df_temp[f'{formula_name}_bins'] = pd.cut(df_temp[formula_column], bins=5)
                value_counts = df_temp[f'{formula_name}_bins'].value_counts()
                fig = px.pie(values=value_counts.values, names=value_counts.index, 
                           title=f"Distribution of {formula_name}")
        
        elif chart_type == 'histogram':
            fig = px.histogram(df, x=formula_column, 
                             title=f"Distribution of {formula_name}")
        
        elif chart_type == 'box':
            if x_column == formula_column:
                fig = px.box(df, y=formula_column, 
                           title=f"Box Plot of {formula_name}")
            else:
                fig = px.box(df, x=x_column, y=formula_column, 
                           title=f"{formula_name} by {x_column}")
        
        # Enhanced styling
        fig.update_layout(
            title_font_size=18,
            title_font_color='#2c3e50',
            font=dict(family='Inter, sans-serif', size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Inter"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"✅ Chart created for formula: {formula_name}")
        
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")

def create_chart_builder(df):
    """Create advanced chart builder interface."""
    st.subheader("📊 Advanced Chart Builder")
    
    # Chart configuration
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox(
            "Chart Type:",
            ["bar", "line", "scatter", "pie", "heatmap", "box", "violin", "histogram"],
            key="builder_type"
        )
        
        x_column = st.selectbox("X-Axis:", df.columns.tolist(), key="builder_x")
        y_column = st.selectbox("Y-Axis:", df.columns.tolist(), key="builder_y")
        
    with col2:
        color_column = st.selectbox("Color (Optional):", ["None"] + df.columns.tolist(), key="builder_color")
        if color_column == "None":
            color_column = None
            
        size_column = st.selectbox("Size (Optional):", ["None"] + df.columns.tolist(), key="builder_size")
        if size_column == "None":
            size_column = None
    
    chart_title = st.text_input("Chart Title:", f"Custom {chart_type.title()} Chart", key="builder_title")
    
    # Generate chart
    if st.button("🎨 Generate Chart", key="generate_chart"):
        fig = create_safe_chart(df, chart_type, x_column, y_column, color_column, size_column, chart_title)
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Main function that runs the ultimate dashboard."""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Ultimate Upwork Jobs Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Dashboard Controls")
        
        if st.button("🔄 Refresh Data", type="primary"):
            st.session_state.last_refresh = datetime.now() - timedelta(seconds=REFRESH_INTERVAL + 1)
            st.rerun()
        
        st.info(f"⏱️ Auto-refresh every {REFRESH_INTERVAL} seconds")
        
        st.markdown("### 📋 Data Source")
        st.write(f"**Sheet:** {SHEET_NAME}")
        st.write(f"**Worksheet:** {WORKSHEET_NAME}")
        
        # Navigation
        st.markdown("### 🧭 Navigation")
        page = st.selectbox(
            "Select Page:",
            ["📊 Overview", "🧪 Experiments", "🧮 Formulas", "📊 Chart Builder"]
        )
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("❌ Unable to load data. Please check your configuration and credentials.")
        return
    
    # Display refresh information
    last_refresh_str = st.session_state.last_refresh.strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f'<div class="refresh-info">🔄 Last refreshed: {last_refresh_str}</div>', unsafe_allow_html=True)
    
    # Main content based on page selection
    if page == "📊 Overview":
        # Advanced metrics
        create_advanced_metrics(df)
        
        # Quick charts
        st.subheader("📈 Quick Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = create_safe_chart(df, 'bar', 'Category', 'Score', title='Score by Category')
            if fig1:
                st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = create_safe_chart(df, 'pie', 'Country', 'Score', title='Jobs by Country')
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            fig3 = create_safe_chart(df, 'scatter', 'Amount spent', 'Score', 'ICP_Score', title='Score vs Amount (ICP Colored)')
            if fig3:
                st.plotly_chart(fig3, use_container_width=True)
        
        with col4:
            fig4 = create_safe_chart(df, 'box', 'Score_Category', 'Amount spent', title='Amount Distribution by Score')
            if fig4:
                st.plotly_chart(fig4, use_container_width=True)
    
    elif page == "🧪 Experiments":
        create_experiment_interface(df)
    
    elif page == "🧮 Formulas":
        create_formula_builder(df)
    
    elif page == "📊 Chart Builder":
        create_chart_builder(df)
    
    # Data table
    with st.expander("📋 Raw Data Table"):
        selected_columns = st.multiselect(
            "Select columns to display:",
            options=df.columns.tolist(),
            default=['Job Title', 'Category', 'Score', 'Amount spent', 'Proposals', 'Country', 'ICP_Score', 'Quality_Score'],
            help="Choose which columns to display in the table"
        )
        
        if selected_columns:
            st.dataframe(
                df[selected_columns],
                use_container_width=True,
                height=400
            )
        else:
            st.warning("Please select at least one column to display.")
    
    # Auto-refresh mechanism
    if should_refresh_data():
        time.sleep(1)
        st.rerun()

if __name__ == "__main__":
    main()
