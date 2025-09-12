"""
Advanced Visualization Engine for Upwork Jobs Dashboard
Creates beautiful, interactive charts for all analytics

WHY: We need a centralized place to create all visualizations with consistent styling
and proper error handling. This makes the dashboard look professional and user-friendly.

HOW: We use Plotly for interactive charts and create reusable functions for each
chart type, ensuring consistent styling and proper data handling.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import numpy as np

class UpworkVisualizationEngine:
    """
    Advanced visualization engine for creating interactive charts.
    
    WHY: We need a centralized place to create all visualizations with consistent styling.
    This class provides a clean interface for all chart creation.
    
    HOW: Each method creates a specific type of chart with proper styling,
    error handling, and interactive features.
    """
    
    def __init__(self):
        """Initialize the visualization engine with default styling."""
        self.color_palette = px.colors.qualitative.Set3
        self.sequential_palette = px.colors.sequential.Blues
    
    def create_weekly_jobs_chart(self, weekly_data: pd.DataFrame) -> go.Figure:
        """
        Create a line chart for weekly job counts.
        
        WHY: Line charts are perfect for showing trends over time.
        This helps visualize job scraping performance week by week.
        
        HOW: We use Plotly's line chart with markers to show weekly job counts,
        adding hover information and proper styling.
        """
        if weekly_data.empty:
            return self._create_empty_chart("No weekly data available")
        
        fig = go.Figure()
        
        # Add line trace
        fig.add_trace(go.Scatter(
            x=weekly_data['Week'],
            y=weekly_data['Job_Count'],
            mode='lines+markers',
            name='Jobs Scraped',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8, color='#667eea'),
            hovertemplate='<b>%{x}</b><br>Jobs: %{y}<br>%{text}<extra></extra>',
            text=[f"{row['Week_Start'].strftime('%m/%d')} - {row['Week_End'].strftime('%m/%d')}" 
                  for _, row in weekly_data.iterrows()]
        ))
        
        # Add trend line
        if len(weekly_data) > 1:
            z = np.polyfit(range(len(weekly_data)), weekly_data['Job_Count'], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=weekly_data['Week'],
                y=p(range(len(weekly_data))),
                mode='lines',
                name='Trend',
                line=dict(color='#764ba2', width=2, dash='dash'),
                opacity=0.7
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': '📊 Weekly Job Scraping Performance',
                'x': 0.5,
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            xaxis_title='Week',
            yaxis_title='Number of Jobs',
            hovermode='x unified',
            template='plotly_white',
            font=dict(family='Inter, sans-serif', size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        
        return fig
    
    def create_daily_jobs_chart(self, daily_data: pd.DataFrame) -> go.Figure:
        """
        Create a bar chart for daily job counts.
        
        WHY: Bar charts are great for showing daily variations and patterns.
        This helps identify busy days and trends in job posting activity.
        
        HOW: We use Plotly's bar chart with color coding based on job count,
        making it easy to spot high and low activity days.
        """
        if daily_data.empty:
            return self._create_empty_chart("No daily data available")
        
        # Add color based on job count
        daily_data['Color'] = daily_data['Job_Count'].apply(
            lambda x: '#2ecc71' if x > daily_data['Job_Count'].mean() else '#e74c3c'
        )
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=daily_data['Date'],
            y=daily_data['Job_Count'],
            marker_color=daily_data['Color'],
            name='Daily Jobs',
            hovertemplate='<b>%{x}</b><br>Jobs: %{y}<extra></extra>'
        ))
        
        # Add average line
        avg_jobs = daily_data['Job_Count'].mean()
        fig.add_hline(
            y=avg_jobs,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Average: {avg_jobs:.1f}",
            annotation_position="top right"
        )
        
        fig.update_layout(
            title={
                'text': '📅 Daily Job Scraping Activity',
                'x': 0.5,
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            xaxis_title='Date',
            yaxis_title='Number of Jobs',
            template='plotly_white',
            font=dict(family='Inter, sans-serif', size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        
        return fig
    
    def create_hourly_distribution_chart(self, hourly_data: pd.DataFrame) -> go.Figure:
        """
        Create a bar chart for hourly distribution of job postings.
        
        WHY: Understanding when jobs are posted helps optimize application timing.
        This chart reveals patterns in job posting behavior throughout the day.
        
        HOW: We use a bar chart with 24 hours on the x-axis, color-coded by time of day
        (morning, afternoon, evening, night) for better visual understanding.
        """
        if hourly_data.empty:
            return self._create_empty_chart("No hourly data available")
        
        # Categorize hours
        def get_time_category(hour):
            if 6 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 18:
                return 'Afternoon'
            elif 18 <= hour < 24:
                return 'Evening'
            else:
                return 'Night'
        
        hourly_data['Time_Category'] = hourly_data['Hour'].apply(get_time_category)
        
        # Color mapping
        color_map = {
            'Morning': '#f39c12',
            'Afternoon': '#2ecc71',
            'Evening': '#e74c3c',
            'Night': '#9b59b6'
        }
        
        hourly_data['Color'] = hourly_data['Time_Category'].map(color_map)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=hourly_data['Time_Label'],
            y=hourly_data['Job_Count'],
            marker_color=hourly_data['Color'],
            name='Jobs Posted',
            hovertemplate='<b>%{x}</b><br>Jobs: %{y}<br>%{text}<extra></extra>',
            text=hourly_data['Time_Category']
        ))
        
        fig.update_layout(
            title={
                'text': '🕐 Hourly Distribution of Job Postings',
                'x': 0.5,
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            xaxis_title='Hour of Day',
            yaxis_title='Number of Jobs',
            template='plotly_white',
            font=dict(family='Inter, sans-serif', size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            xaxis=dict(tickangle=45)
        )
        
        return fig
    
    def create_status_breakdown_chart(self, status_data: pd.DataFrame) -> go.Figure:
        """
        Create a stacked bar chart for weekly status breakdown.
        
        WHY: Stacked bars show both individual status counts and total volume.
        This helps track conversion rates and status trends over time.
        
        HOW: We create a stacked bar chart with different colors for each status,
        making it easy to compare status distributions across weeks.
        """
        if status_data.empty:
            return self._create_empty_chart("No status data available")
        
        # Get status columns (excluding week info columns)
        status_columns = [col for col in status_data.columns if col.startswith('Status_')]
        
        if not status_columns:
            return self._create_empty_chart("No status columns found")
        
        fig = go.Figure()
        
        # Color palette for different statuses
        colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6', '#1abc9c']
        
        for i, col in enumerate(status_columns):
            status_name = col.replace('Status_', '')
            fig.add_trace(go.Bar(
                name=status_name,
                x=status_data['Week'],
                y=status_data[col],
                marker_color=colors[i % len(colors)],
                hovertemplate=f'<b>{status_name}</b><br>Week: %{{x}}<br>Count: %{{y}}<extra></extra>'
            ))
        
        fig.update_layout(
            barmode='stack',
            title={
                'text': '📊 Weekly Application Status Breakdown',
                'x': 0.5,
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            xaxis_title='Week',
            yaxis_title='Number of Jobs',
            template='plotly_white',
            font=dict(family='Inter, sans-serif', size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        
        return fig
    
    def create_icp_analysis_chart(self, icp_data: Dict[str, Any]) -> go.Figure:
        """
        Create charts for ICP analysis.
        
        WHY: ICP analysis is crucial for identifying high-value opportunities.
        This visualization helps understand job quality and targeting effectiveness.
        
        HOW: We create multiple subplots showing ICP distribution, high-value jobs,
        and average scores with clear visual indicators.
        """
        if "error" in icp_data:
            return self._create_empty_chart(icp_data["error"])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ICP Score Distribution', 'High ICP Jobs', 
                          'Average ICP Score', 'ICP Performance'),
            specs=[[{"type": "bar"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "pie"}]]
        )
        
        # ICP Score Distribution
        if "icp_distribution" in icp_data:
            scores = list(icp_data["icp_distribution"].keys())
            counts = list(icp_data["icp_distribution"].values())
            
            fig.add_trace(
                go.Bar(x=scores, y=counts, name='ICP Distribution', marker_color='#3498db'),
                row=1, col=1
            )
        
        # High ICP Jobs (Gauge)
        high_icp = icp_data.get("high_icp_jobs", 0)
        total_icp = icp_data.get("total_icp_jobs", 1)
        high_icp_pct = (high_icp / total_icp) * 100 if total_icp > 0 else 0
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=high_icp_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "High ICP Jobs %"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}]}
            ),
            row=1, col=2
        )
        
        # Average ICP Score (Gauge)
        avg_score = icp_data.get("avg_icp_score", 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=avg_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Average ICP Score"},
                gauge={'axis': {'range': [0, 5]},
                       'bar': {'color': "darkgreen"},
                       'steps': [{'range': [0, 2], 'color': "lightgray"},
                                {'range': [2, 3], 'color': "yellow"},
                                {'range': [3, 5], 'color': "green"}]}
            ),
            row=2, col=1
        )
        
        # ICP Performance Pie Chart
        high_icp = icp_data.get("high_icp_jobs", 0)
        low_icp = total_icp - high_icp
        
        fig.add_trace(
            go.Pie(
                labels=['High ICP', 'Low ICP'],
                values=[high_icp, low_icp],
                marker_colors=['#2ecc71', '#e74c3c'],
                name="ICP Performance"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title={
                'text': '🎯 ICP (Ideal Customer Profile) Analysis',
                'x': 0.5,
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            height=600,
            showlegend=False,
            template='plotly_white',
            font=dict(family='Inter, sans-serif', size=12)
        )
        
        return fig
    
    def create_experience_level_chart(self, exp_data: pd.DataFrame) -> go.Figure:
        """
        Create a pie chart for experience level distribution.
        
        WHY: Pie charts are perfect for showing proportional distributions.
        This helps understand the market composition by experience level.
        
        HOW: We use a pie chart with custom colors and hover information
        to show the distribution of jobs by experience level.
        """
        if exp_data.empty:
            return self._create_empty_chart("No experience level data available")
        
        fig = go.Figure()
        
        fig.add_trace(go.Pie(
            labels=exp_data['Experience_Level'],
            values=exp_data['Count'],
            hole=0.4,
            marker_colors=['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6'],
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': '👥 Experience Level Distribution',
                'x': 0.5,
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            template='plotly_white',
            font=dict(family='Inter, sans-serif', size=12),
            height=400
        )
        
        return fig
    
    def create_hourly_rate_chart(self, rate_data: Dict[str, Any]) -> go.Figure:
        """
        Create charts for hourly rate analysis.
        
        WHY: Rate analysis helps understand market pricing and identify opportunities.
        This visualization shows rate distributions and high-value job opportunities.
        
        HOW: We create multiple subplots showing rate ranges, average rates,
        and high-paying job opportunities with clear visual indicators.
        """
        if "error" in rate_data:
            return self._create_empty_chart(rate_data["error"])
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Hourly Rate Ranges', 'Rate Analysis'),
            specs=[[{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # Rate Ranges Bar Chart
        if "rate_ranges" in rate_data:
            ranges = list(rate_data["rate_ranges"].keys())
            counts = list(rate_data["rate_ranges"].values())
            
            # Color based on rate level
            colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
            
            fig.add_trace(
                go.Bar(
                    x=ranges,
                    y=counts,
                    name='Rate Ranges',
                    marker_color=colors,
                    hovertemplate='<b>%{x}</b><br>Jobs: %{y}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Average Rate Gauge
        avg_rate = rate_data.get("avg_hourly_rate", 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=avg_rate,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Average Hourly Rate ($)"},
                gauge={'axis': {'range': [0, 200]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 25], 'color': "lightgray"},
                                {'range': [25, 50], 'color': "yellow"},
                                {'range': [50, 100], 'color': "orange"},
                                {'range': [100, 200], 'color': "green"}]}
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title={
                'text': '💰 Hourly Rate Analysis',
                'x': 0.5,
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            height=400,
            showlegend=False,
            template='plotly_white',
            font=dict(family='Inter, sans-serif', size=12)
        )
        
        return fig
    
    def create_country_analysis_chart(self, country_data: pd.DataFrame) -> go.Figure:
        """
        Create a horizontal bar chart for country analysis.
        
        WHY: Horizontal bars are great for country names and make comparison easier.
        This helps identify the top markets and geographic distribution.
        
        HOW: We use a horizontal bar chart with color coding based on job count,
        making it easy to compare countries and identify top markets.
        """
        if country_data.empty:
            return self._create_empty_chart("No country data available")
        
        # Sort by job count for better visualization
        country_data = country_data.sort_values('Job_Count', ascending=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=country_data['Country'],
            x=country_data['Job_Count'],
            orientation='h',
            marker_color='#3498db',
            hovertemplate='<b>%{y}</b><br>Jobs: %{x}<br>Percentage: %{text}<extra></extra>',
            text=[f"{pct:.1f}%" for pct in country_data['Percentage']]
        ))
        
        fig.update_layout(
            title={
                'text': '🌍 Top Countries by Job Count',
                'x': 0.5,
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            xaxis_title='Number of Jobs',
            yaxis_title='Country',
            template='plotly_white',
            font=dict(family='Inter, sans-serif', size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        
        return fig
    
    def create_real_time_metrics_display(self, metrics: Dict[str, Any]) -> None:
        """
        Create a real-time metrics display using Streamlit components.
        
        WHY: Real-time metrics need to be prominently displayed for quick insights.
        This provides immediate visibility into current performance.
        
        HOW: We use Streamlit's metric and columns components to create
        a dashboard-style display of key performance indicators.
        """
        st.subheader("📊 Real-Time Dashboard Metrics")
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Jobs",
                value=f"{metrics.get('total_jobs', 0):,}",
                delta=f"+{metrics.get('today_jobs', 0)} today"
            )
        
        with col2:
            st.metric(
                label="Application Rate",
                value=f"{metrics.get('application_rate', 0):.1f}%",
                delta=f"{metrics.get('applied_jobs', 0)} applied"
            )
        
        with col3:
            st.metric(
                label="This Week",
                value=f"{metrics.get('week_jobs', 0):,}",
                delta=f"{metrics.get('avg_daily_rate', 0):.1f} avg/day"
            )
        
        with col4:
            st.metric(
                label="High Value Jobs",
                value=f"{metrics.get('high_value_jobs', 0):,}",
                delta="ICP + Rate"
            )
        
        # Additional metrics
        col5, col6, col7 = st.columns(3)
        
        with col5:
            st.metric(
                label="This Month",
                value=f"{metrics.get('month_jobs', 0):,}",
                delta="Current month"
            )
        
        with col6:
            st.metric(
                label="Last Updated",
                value=metrics.get('last_updated', 'Unknown'),
                delta=""
            )
        
        with col7:
            st.metric(
                label="Today's Jobs",
                value=f"{metrics.get('today_jobs', 0):,}",
                delta="Today only"
            )
    
    def create_conversion_funnel_chart(self, funnel_data: Dict[str, int]) -> go.Figure:
        """
        Create a funnel chart for conversion analysis.
        
        WHY: Funnel charts are perfect for showing conversion rates through stages.
        This helps identify bottlenecks in the job application process.
        
        HOW: We use Plotly's funnel chart to show the flow from total scraped
        to passed filters to applied, with clear visual indicators.
        """
        stages = ['Total Scraped', 'Passed Filters', 'Applied']
        values = [
            funnel_data.get('total_scraped', 0),
            funnel_data.get('passed_filters', 0),
            funnel_data.get('applied', 0)
        ]
        
        fig = go.Figure(go.Funnel(
            y=stages,
            x=values,
            textinfo="value+percent initial",
            marker={"color": ["#e74c3c", "#f39c12", "#2ecc71"]},
            connector={"line": {"color": "royalblue", "dash": "dot", "width": 3}}
        ))
        
        fig.update_layout(
            title={
                'text': '🔄 Job Application Conversion Funnel',
                'x': 0.5,
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            template='plotly_white',
            font=dict(family='Inter, sans-serif', size=12),
            height=400
        )
        
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """
        Create an empty chart with a message.
        
        WHY: We need to handle cases where no data is available gracefully.
        
        HOW: We create a simple figure with text annotation showing the message.
        """
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font_size=16, font_color="gray"
        )
        fig.update_layout(
            xaxis={'visible': False},
            yaxis={'visible': False},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=300
        )
        return fig
