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
from analytics_engine import UpworkAnalyticsEngine
from visualization_engine import UpworkVisualizationEngine
from proposal_tracking_engine import ProposalTrackingEngine
from proposal_visualization_engine import ProposalVisualizationEngine
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
        
        credentials_path = "service_account_credentials.json"
        
        if not os.path.exists(credentials_path):
            st.error(f"❌ Service account credentials file not found: {credentials_path}")
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

def load_proposal_data():
    """Load proposal tracking data from Google Sheets."""
    try:
        if (st.session_state.data_cache is not None and 
            st.session_state.cache_timestamp is not None and 
            not should_refresh_data()):
            return st.session_state.data_cache
        
        credentials_path = "service_account_credentials.json"
        
        if not os.path.exists(credentials_path):
            st.error(f"❌ Service account credentials file not found: {credentials_path}")
            return None
        
        connector = GoogleSheetsConnector(credentials_path)
        df = connector.get_sheet_data(SHEET_ID, PROPOSAL_TRACKING_SHEET)
        
        if df is not None:
            st.session_state.data_cache = df
            st.session_state.cache_timestamp = datetime.now()
            st.session_state.last_refresh = datetime.now()
            return df
        else:
            st.error("❌ Failed to load proposal tracking data from Google Sheets")
            return None
            
    except Exception as e:
        st.error(f"❌ Error loading proposal data: {str(e)}")
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Create New Formula")
        
        formula_name = st.text_input("Formula Name:", key="formula_name")
        
        formula_expression = st.text_area(
            "Formula Expression (use column names):",
            placeholder="Example: (Score * 0.4) + (Amount spent / 1000 * 0.3) + (Proposals * 0.2)",
            key="formula_expr"
        )
        
        if st.button("🔧 Create Formula", key="create_formula"):
            if formula_name and formula_expression:
                create_custom_formula(df, formula_name, formula_expression)
            else:
                st.warning("Please fill in all fields")
    
    with col2:
        st.markdown("### Saved Formulas")
        
        if st.session_state.custom_formulas:
            for formula_name, formula_data in st.session_state.custom_formulas.items():
                with st.expander(f"📐 {formula_name}"):
                    st.code(formula_data['expression'])
                    if st.button(f"Apply {formula_name}", key=f"apply_{formula_name}"):
                        apply_formula(df, formula_name, formula_data['expression'])

def create_custom_formula(df, name, expression):
    """Create and apply custom formula."""
    try:
        # Validate expression by trying to evaluate it
        test_result = df.eval(expression, engine='python')
        
        # Store formula
        st.session_state.custom_formulas[name] = {
            'expression': expression,
            'timestamp': datetime.now().isoformat()
        }
        
        # Apply formula
        df[f'Custom_{name}'] = test_result
        
        st.success(f"✅ Formula '{name}' created and applied!")
        st.info(f"New column 'Custom_{name}' added to dataset")
        
    except Exception as e:
        st.error(f"Error creating formula: {str(e)}")

def apply_formula(df, name, expression):
    """Apply existing formula to dataset."""
    try:
        df[f'Custom_{name}'] = df.eval(expression, engine='python')
        st.success(f"✅ Formula '{name}' applied!")
    except Exception as e:
        st.error(f"Error applying formula: {str(e)}")

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
            ["📊 Overview", "📈 Time Analytics", "📊 Status Analysis", "🎯 Advanced Analytics", "💼 Proposal Tracking", "🧪 Experiments", "🧮 Formulas", "📊 Chart Builder"]
        )
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("❌ Unable to load data. Please check your configuration and credentials.")
        return
    
    # Display refresh information
    last_refresh_str = st.session_state.last_refresh.strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f'<div class="refresh-info">🔄 Last refreshed: {last_refresh_str}</div>', unsafe_allow_html=True)
    
    # Initialize analytics engines
    analytics_engine = UpworkAnalyticsEngine(df)
    viz_engine = UpworkVisualizationEngine()
    
    # Load proposal tracking data if needed
    proposal_df = None
    if page == "💼 Proposal Tracking":
        proposal_df = load_proposal_data()
        if proposal_df is not None:
            proposal_engine = ProposalTrackingEngine(proposal_df)
            proposal_viz_engine = ProposalVisualizationEngine()
    
    # Main content based on page selection
    if page == "📊 Overview":
        # Real-time metrics
        real_time_metrics = analytics_engine.get_real_time_metrics()
        viz_engine.create_real_time_metrics_display(real_time_metrics)
        
        # Conversion funnel
        st.subheader("🔄 Conversion Funnel")
        funnel_data = analytics_engine.get_conversion_funnel()
        funnel_chart = viz_engine.create_conversion_funnel_chart(funnel_data)
        st.plotly_chart(funnel_chart, use_container_width=True)
        
        # Quick charts
        st.subheader("📈 Quick Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = create_safe_chart(df, 'bar', 'Category', 'Score', title='Score by Category')
            if fig1:
                st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            country_data = analytics_engine.get_country_analysis(top_n=10)
            if not country_data.empty:
                country_chart = viz_engine.create_country_analysis_chart(country_data)
                st.plotly_chart(country_chart, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            fig3 = create_safe_chart(df, 'scatter', 'Amount spent', 'Score', 'ICP_Score', title='Score vs Amount (ICP Colored)')
            if fig3:
                st.plotly_chart(fig3, use_container_width=True)
        
        with col4:
            fig4 = create_safe_chart(df, 'box', 'Score_Category', 'Amount spent', title='Amount Distribution by Score')
            if fig4:
                st.plotly_chart(fig4, use_container_width=True)
    
    elif page == "📈 Time Analytics":
        st.subheader("📈 Time-Based Analytics")
        
        # Weekly job counts
        st.markdown("### 📊 Weekly Job Scraping Performance")
        weekly_data = analytics_engine.get_weekly_job_counts()
        if not weekly_data.empty:
            weekly_chart = viz_engine.create_weekly_jobs_chart(weekly_data)
            st.plotly_chart(weekly_chart, use_container_width=True)
        else:
            st.warning("No weekly data available. Please check your Publish Date column.")
        
        # Daily job counts
        st.markdown("### 📅 Daily Job Scraping Activity")
        daily_data = analytics_engine.get_daily_job_counts()
        if not daily_data.empty:
            daily_chart = viz_engine.create_daily_jobs_chart(daily_data)
            st.plotly_chart(daily_chart, use_container_width=True)
        else:
            st.warning("No daily data available. Please check your Publish Date column.")
        
        # Hourly distribution
        st.markdown("### 🕐 Hourly Distribution of Job Postings")
        hourly_data = analytics_engine.get_hourly_distribution()
        if not hourly_data.empty:
            hourly_chart = viz_engine.create_hourly_distribution_chart(hourly_data)
            st.plotly_chart(hourly_chart, use_container_width=True)
        else:
            st.warning("No hourly data available. Please check your Publish Date column.")
    
    elif page == "📊 Status Analysis":
        st.subheader("📊 Application Status Analysis")
        
        # Weekly status breakdown
        st.markdown("### 📊 Weekly Application Status Breakdown")
        status_data = analytics_engine.get_status_breakdown_weekly()
        if not status_data.empty:
            status_chart = viz_engine.create_status_breakdown_chart(status_data)
            st.plotly_chart(status_chart, use_container_width=True)
        else:
            st.warning("No status data available. Please check your Application Status column.")
        
        # Status summary
        if COLUMNS['application_status_column'] in df.columns:
            st.markdown("### 📋 Status Summary")
            status_summary = df[COLUMNS['application_status_column']].value_counts()
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(status_summary.reset_index().rename(columns={'index': 'Status', COLUMNS['application_status_column']: 'Count'}))
            
            with col2:
                # Status pie chart
                fig = px.pie(
                    values=status_summary.values,
                    names=status_summary.index,
                    title="Status Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🎯 Advanced Analytics":
        st.subheader("🎯 Advanced Analytics")
        
        # ICP Analysis
        st.markdown("### 🎯 ICP (Ideal Customer Profile) Analysis")
        icp_data = analytics_engine.get_icp_analysis()
        if "error" not in icp_data:
            icp_chart = viz_engine.create_icp_analysis_chart(icp_data)
            st.plotly_chart(icp_chart, use_container_width=True)
        else:
            st.warning(f"ICP Analysis: {icp_data['error']}")
        
        # Experience Level Analysis
        st.markdown("### 👥 Experience Level Distribution")
        exp_data = analytics_engine.get_experience_level_distribution()
        if not exp_data.empty:
            exp_chart = viz_engine.create_experience_level_chart(exp_data)
            st.plotly_chart(exp_chart, use_container_width=True)
        else:
            st.warning("No experience level data available.")
        
        # Hourly Rate Analysis
        st.markdown("### 💰 Hourly Rate Analysis")
        rate_data = analytics_engine.get_hourly_rate_analysis()
        if "error" not in rate_data:
            rate_chart = viz_engine.create_hourly_rate_chart(rate_data)
            st.plotly_chart(rate_chart, use_container_width=True)
        else:
            st.warning(f"Hourly Rate Analysis: {rate_data['error']}")
        
        # Country Analysis
        st.markdown("### 🌍 Geographic Analysis")
        country_data = analytics_engine.get_country_analysis(top_n=15)
        if not country_data.empty:
            country_chart = viz_engine.create_country_analysis_chart(country_data)
            st.plotly_chart(country_chart, use_container_width=True)
        else:
            st.warning("No country data available.")
    
    elif page == "💼 Proposal Tracking":
        if proposal_df is None:
            st.error("❌ Unable to load proposal tracking data. Please check your configuration and credentials.")
        else:
            st.subheader("💼 Proposal Tracking Dashboard")
            
            # Get all proposal tracking data
            baseline_data = proposal_engine.get_baseline_2025_data()
            outbound_goals = proposal_engine.get_outbound_monthly_goals()
            inbound_goals = proposal_engine.get_inbound_monthly_goals()
            daily_targets = proposal_engine.get_daily_activity_targets()
            performance_data = proposal_engine.get_performance_tracking()
            advanced_data = proposal_engine.get_advanced_analytics()
            dashboard_summary = proposal_engine.get_dashboard_summary()
            
            # Baseline 2025 Data Section
            st.markdown("### 📊 Baseline 2025 Data")
            proposal_viz_engine.create_baseline_metrics_display(baseline_data)
            
            # Conversion Funnel
            st.markdown("### 🔄 Conversion Funnel")
            funnel_chart = proposal_viz_engine.create_conversion_funnel_chart(baseline_data)
            st.plotly_chart(funnel_chart, use_container_width=True)
            
            # Outbound and Inbound Goals
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎯 Outbound Monthly Goals (90%)")
                outbound_chart = proposal_viz_engine.create_outbound_goals_chart(outbound_goals)
                st.plotly_chart(outbound_chart, use_container_width=True)
            
            with col2:
                st.markdown("### 📈 Inbound Monthly Goals (90%)")
                inbound_chart = proposal_viz_engine.create_inbound_goals_chart(inbound_goals)
                st.plotly_chart(inbound_chart, use_container_width=True)
            
            # Daily Activity Targets
            st.markdown("### 📅 Daily Activity Targets")
            daily_chart = proposal_viz_engine.create_daily_targets_chart(daily_targets)
            st.plotly_chart(daily_chart, use_container_width=True)
            
            # Performance Tracking
            st.markdown("### 📊 Performance Tracking (Actual vs Target)")
            performance_chart = proposal_viz_engine.create_performance_tracking_chart(performance_data)
            st.plotly_chart(performance_chart, use_container_width=True)
            
            # Advanced Analytics
            st.markdown("### 🚀 Advanced Analytics & Forecasting")
            advanced_chart = proposal_viz_engine.create_advanced_analytics_chart(advanced_data)
            st.plotly_chart(advanced_chart, use_container_width=True)
            
            # Financial Summary
            proposal_viz_engine.create_financial_summary_display(dashboard_summary)
            
            # Data table
            with st.expander("📋 Proposal Tracking Data Table"):
                st.dataframe(
                    proposal_df,
                    use_container_width=True,
                    height=400
                )
    
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
