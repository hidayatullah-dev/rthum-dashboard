"""
Proposal Tracking Visualization Engine
Creates beautiful, interactive charts for proposal tracking analytics

WHY: We need a centralized place to create all proposal tracking visualizations with
consistent styling and proper error handling. This makes the dashboard look professional.

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

class ProposalVisualizationEngine:
    """
    Advanced visualization engine for proposal tracking charts.
    
    WHY: We need a centralized place to create all proposal tracking visualizations.
    This class provides a clean interface for all chart creation.
    
    HOW: Each method creates a specific type of chart with proper styling,
    error handling, and interactive features.
    """
    
    def __init__(self):
        """Initialize the visualization engine with default styling."""
        self.color_palette = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6', '#1abc9c']
        self.sequential_palette = px.colors.sequential.Blues
    
    def create_baseline_metrics_display(self, baseline_data: Dict[str, Any]) -> None:
        """
        Create baseline 2025 metrics display using Streamlit components.
        
        WHY: Baseline metrics need to be prominently displayed for quick insights.
        This provides immediate visibility into current performance.
        
        HOW: We use Streamlit's metric and columns components to create
        a dashboard-style display of key performance indicators.
        """
        st.subheader("📊 Baseline 2025 Data")
        
        # Core metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Applications Sent",
                value=f"{baseline_data.get('applications_sent', 0):,}",
                delta=""
            )
        
        with col2:
            st.metric(
                label="Replies",
                value=f"{baseline_data.get('replies', 0):,}",
                delta=f"{baseline_data.get('reply_conversion_rate', 0):.1f}% conversion"
            )
        
        with col3:
            st.metric(
                label="Interviews",
                value=f"{baseline_data.get('interviews', 0):,}",
                delta=f"{baseline_data.get('interview_conversion_rate', 0):.1f}% conversion"
            )
        
        with col4:
            st.metric(
                label="Jobs Won",
                value=f"{baseline_data.get('jobs_won', 0):,}",
                delta=f"{baseline_data.get('job_won_conversion_rate', 0):.1f}% conversion"
            )
        
        # Financial metrics
        col5, col6, col7 = st.columns(3)
        
        with col5:
            st.metric(
                label="Overall Hire Rate",
                value=f"{baseline_data.get('overall_hire_rate', 0):.1f}%",
                delta="Applications to Hires"
            )
        
        with col6:
            st.metric(
                label="Monthly Revenue",
                value=f"${baseline_data.get('monthly_gross_revenue', 0):,.0f}",
                delta=f"${baseline_data.get('average_deal_size', 0):,.0f} avg deal"
            )
        
        with col7:
            st.metric(
                label="LTV Calculation",
                value=f"${baseline_data.get('ltv_calculation', 0):,.0f}",
                delta=f"{baseline_data.get('ltv_multiplier', 0)}x multiplier"
            )
    
    def create_conversion_funnel_chart(self, baseline_data: Dict[str, Any]) -> go.Figure:
        """
        Create a conversion funnel chart for the proposal process.
        
        WHY: Funnel charts are perfect for showing conversion rates through stages.
        This helps identify bottlenecks in the proposal process.
        
        HOW: We use Plotly's funnel chart to show the flow from applications
        to replies to interviews to jobs won, with clear visual indicators.
        """
        stages = ['Applications Sent', 'Replies', 'Interviews', 'Jobs Won']
        values = [
            baseline_data.get('applications_sent', 0),
            baseline_data.get('replies', 0),
            baseline_data.get('interviews', 0),
            baseline_data.get('jobs_won', 0)
        ]
        
        # Calculate conversion rates for display
        conversion_rates = []
        for i in range(1, len(values)):
            if values[i-1] > 0:
                rate = (values[i] / values[i-1]) * 100
                conversion_rates.append(f"{rate:.1f}%")
            else:
                conversion_rates.append("0%")
        
        fig = go.Figure(go.Funnel(
            y=stages,
            x=values,
            textinfo="value+percent initial",
            marker={"color": self.color_palette[:len(stages)]},
            connector={"line": {"color": "royalblue", "dash": "dot", "width": 3}}
        ))
        
        # Add conversion rate annotations
        for i, (stage, value, rate) in enumerate(zip(stages[1:], values[1:], conversion_rates)):
            fig.add_annotation(
                x=value,
                y=stage,
                text=f"<b>{rate}</b>",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red",
                ax=20,
                ay=-30,
                font=dict(size=12, color="red")
            )
        
        fig.update_layout(
            title={
                'text': '🔄 Proposal Conversion Funnel',
                'x': 0.5,
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            template='plotly_white',
            font=dict(family='Inter, sans-serif', size=12),
            height=500
        )
        
        return fig
    
    def create_outbound_goals_chart(self, outbound_data: Dict[str, Any]) -> go.Figure:
        """
        Create a chart showing outbound goals vs targets.
        
        WHY: This helps visualize progress against outbound targets.
        It shows both actual performance and target goals.
        
        HOW: We create a grouped bar chart comparing actual vs target metrics
        with clear visual indicators for performance.
        """
        metrics = ['Applications', 'Replies', 'Interviews', 'Jobs Won']
        targets = [
            outbound_data.get('target_applications', 0),
            outbound_data.get('target_replies', 0),
            outbound_data.get('target_interviews', 0),
            outbound_data.get('target_jobs_won', 0)
        ]
        
        # For now, we'll show targets only. In a real implementation,
        # you'd want to compare with actual performance
        actuals = [0, 0, 0, 0]  # Placeholder for actual data
        
        fig = go.Figure()
        
        # Add target bars
        fig.add_trace(go.Bar(
            name='Target',
            x=metrics,
            y=targets,
            marker_color='#3498db',
            hovertemplate='<b>%{x}</b><br>Target: %{y}<br><extra></extra>'
        ))
        
        # Add actual bars (if we had actual data)
        if any(actuals):
            fig.add_trace(go.Bar(
                name='Actual',
                x=metrics,
                y=actuals,
                marker_color='#2ecc71',
                hovertemplate='<b>%{x}</b><br>Actual: %{y}<br><extra></extra>'
            ))
        
        fig.update_layout(
            title={
                'text': '🎯 Outbound Monthly Goals (90%)',
                'x': 0.5,
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            xaxis_title='Metrics',
            yaxis_title='Count',
            template='plotly_white',
            font=dict(family='Inter, sans-serif', size=12),
            height=400,
            barmode='group'
        )
        
        return fig
    
    def create_inbound_goals_chart(self, inbound_data: Dict[str, Any]) -> go.Figure:
        """
        Create a chart showing inbound goals and conversion rates.
        
        WHY: This visualizes the inbound marketing funnel and conversion rates.
        It helps understand how well your profile visibility is performing.
        
        HOW: We create a multi-panel chart showing both volume metrics
        and conversion rates for the inbound funnel.
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Inbound Funnel Volume', 'Conversion Rates', 
                          'Overachievement Targets', 'Daily Breakdown'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Inbound Funnel Volume
        stages = ['Impressions', 'Profile Views', 'Invites', 'Hires']
        values = [
            inbound_data.get('target_impressions', 0),
            inbound_data.get('target_profile_views', 0),
            inbound_data.get('target_invites', 0),
            inbound_data.get('target_hires', 0)
        ]
        
        fig.add_trace(
            go.Bar(x=stages, y=values, name='Volume', marker_color='#e74c3c'),
            row=1, col=1
        )
        
        # Conversion Rates
        conversion_stages = ['Impressions→Views', 'Views→Invites', 'Invites→Hires']
        conversion_values = [
            inbound_data.get('profile_view_conversion_rate', 0),
            inbound_data.get('invite_conversion_rate', 0),
            inbound_data.get('hire_conversion_rate', 0)
        ]
        
        fig.add_trace(
            go.Bar(x=conversion_stages, y=conversion_values, name='Conversion %', marker_color='#2ecc71'),
            row=1, col=2
        )
        
        # Overachievement Targets
        overachieve_values = [
            inbound_data.get('overachieve_profile_views', 0),
            inbound_data.get('overachieve_invites', 0),
            inbound_data.get('overachieve_hires', 0)
        ]
        
        fig.add_trace(
            go.Bar(x=stages[1:], y=overachieve_values, name='Overachieve', marker_color='#f39c12'),
            row=2, col=1
        )
        
        # Daily Breakdown
        daily_metrics = ['Daily Impressions', 'Daily Views', 'Weekly Invites', 'Monthly Hires']
        daily_values = [
            inbound_data.get('impressions_per_day', 0),
            inbound_data.get('daily_profile_views', 0),
            inbound_data.get('weekly_invites', 0),
            inbound_data.get('monthly_hires', 0)
        ]
        
        fig.add_trace(
            go.Bar(x=daily_metrics, y=daily_values, name='Daily Targets', marker_color='#9b59b6'),
            row=2, col=2
        )
        
        fig.update_layout(
            title={
                'text': '📈 Inbound Monthly Goals (90%)',
                'x': 0.5,
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            height=600,
            showlegend=False,
            template='plotly_white',
            font=dict(family='Inter, sans-serif', size=12)
        )
        
        return fig
    
    def create_daily_targets_chart(self, daily_data: Dict[str, Any]) -> go.Figure:
        """
        Create a chart showing daily activity targets.
        
        WHY: Daily targets help with daily planning and progress tracking.
        This visualization makes it easy to see what needs to be done each day.
        
        HOW: We create a grouped bar chart showing outbound and inbound
        daily targets for easy comparison.
        """
        # Outbound daily targets
        outbound_metrics = ['Applications', 'Replies', 'Interviews', 'Jobs Won']
        outbound_values = [
            daily_data.get('daily_applications', 0),
            daily_data.get('daily_replies', 0),
            daily_data.get('daily_interviews', 0),
            daily_data.get('daily_jobs_won', 0)
        ]
        
        # Inbound daily targets
        inbound_metrics = ['Impressions', 'Profile Views', 'Weekly Invites', 'Monthly Hires']
        inbound_values = [
            daily_data.get('daily_impressions', 0),
            daily_data.get('daily_profile_views', 0),
            daily_data.get('weekly_invites', 0),
            daily_data.get('monthly_hires', 0)
        ]
        
        fig = go.Figure()
        
        # Add outbound bars
        fig.add_trace(go.Bar(
            name='Outbound Daily',
            x=outbound_metrics,
            y=outbound_values,
            marker_color='#3498db',
            hovertemplate='<b>%{x}</b><br>Daily Target: %{y}<br><extra></extra>'
        ))
        
        # Add inbound bars
        fig.add_trace(go.Bar(
            name='Inbound Daily',
            x=inbound_metrics,
            y=inbound_values,
            marker_color='#e74c3c',
            hovertemplate='<b>%{x}</b><br>Daily Target: %{y}<br><extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': '📅 Daily Activity Targets',
                'x': 0.5,
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            xaxis_title='Activity Type',
            yaxis_title='Daily Target',
            template='plotly_white',
            font=dict(family='Inter, sans-serif', size=12),
            height=400,
            barmode='group'
        )
        
        return fig
    
    def create_performance_tracking_chart(self, performance_data: Dict[str, Any]) -> go.Figure:
        """
        Create a chart showing performance vs targets.
        
        WHY: This helps track progress against goals and identify areas needing attention.
        It provides a clear visual comparison of actual vs target performance.
        
        HOW: We create a gauge chart showing performance percentages
        with color coding for different performance levels.
        """
        # Create subplots for different performance metrics
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Outbound Performance', 'Inbound Performance', 'Revenue Performance'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Outbound Performance Gauge
        outbound_perf = performance_data.get('outbound_performance', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=outbound_perf,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Outbound %"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}]}
            ),
            row=1, col=1
        )
        
        # Inbound Performance Gauge
        inbound_perf = performance_data.get('inbound_performance', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=inbound_perf,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Inbound %"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkred"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}]}
            ),
            row=1, col=2
        )
        
        # Revenue Performance Gauge
        revenue_perf = performance_data.get('revenue_performance', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=revenue_perf,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Revenue %"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkgreen"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}]}
            ),
            row=1, col=3
        )
        
        fig.update_layout(
            title={
                'text': '📊 Performance Tracking (Actual vs Target)',
                'x': 0.5,
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            height=400,
            template='plotly_white',
            font=dict(family='Inter, sans-serif', size=12)
        )
        
        return fig
    
    def create_advanced_analytics_chart(self, advanced_data: Dict[str, Any]) -> go.Figure:
        """
        Create a chart showing advanced analytics and forecasting.
        
        WHY: This provides deeper insights into performance trends and future projections.
        It helps with strategic planning and goal setting.
        
        HOW: We create a multi-panel chart showing key metrics, trends,
        and projections for comprehensive analysis.
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Pipeline Value', 'Conversion Rate', 
                          'Revenue Trends', 'Projections'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Pipeline Value Gauge
        pipeline_value = advanced_data.get('total_pipeline_value', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=pipeline_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Pipeline Value ($)"},
                gauge={'axis': {'range': [0, 1000000]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 100000], 'color': "lightgray"},
                                {'range': [100000, 500000], 'color': "yellow"},
                                {'range': [500000, 1000000], 'color': "green"}]}
            ),
            row=1, col=1
        )
        
        # Conversion Rate Gauge
        conversion_rate = advanced_data.get('combined_conversion_rate', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=conversion_rate,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Combined Conversion %"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "darkgreen"},
                       'steps': [{'range': [0, 10], 'color': "lightgray"},
                                {'range': [10, 20], 'color': "yellow"},
                                {'range': [20, 100], 'color': "green"}]}
            ),
            row=1, col=2
        )
        
        # Revenue Trends
        revenue_metrics = ['Current Month', 'Target Month', 'Projected Month']
        revenue_values = [
            advanced_data.get('total_revenue', 0),
            advanced_data.get('total_revenue', 0) * 1.2,  # 20% increase target
            advanced_data.get('end_of_month_projection', 0)
        ]
        
        fig.add_trace(
            go.Bar(x=revenue_metrics, y=revenue_values, name='Revenue', marker_color='#2ecc71'),
            row=2, col=1
        )
        
        # Projections
        projection_metrics = ['Monthly Run Rate', 'End of Month', 'Quarter']
        projection_values = [
            advanced_data.get('monthly_run_rate', 0),
            advanced_data.get('end_of_month_projection', 0),
            advanced_data.get('quarter_projection', 0)
        ]
        
        fig.add_trace(
            go.Bar(x=projection_metrics, y=projection_values, name='Projections', marker_color='#f39c12'),
            row=2, col=2
        )
        
        fig.update_layout(
            title={
                'text': '🚀 Advanced Analytics & Forecasting',
                'x': 0.5,
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            height=600,
            showlegend=False,
            template='plotly_white',
            font=dict(family='Inter, sans-serif', size=12)
        )
        
        return fig
    
    def create_financial_summary_display(self, summary_data: Dict[str, Any]) -> None:
        """
        Create a financial summary display using Streamlit components.
        
        WHY: Financial metrics need to be prominently displayed for quick insights.
        This provides immediate visibility into revenue and financial performance.
        
        HOW: We use Streamlit's metric and columns components to create
        a dashboard-style display of key financial indicators.
        """
        st.subheader("💰 Financial Summary")
        
        baseline = summary_data.get('baseline', {})
        outbound = summary_data.get('outbound_goals', {})
        inbound = summary_data.get('inbound_goals', {})
        advanced = summary_data.get('advanced', {})
        
        # Revenue metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Current Revenue",
                value=f"${baseline.get('monthly_gross_revenue', 0):,.0f}",
                delta="This month"
            )
        
        with col2:
            st.metric(
                label="Target Revenue",
                value=f"${outbound.get('monthly_gross_revenue', 0):,.0f}",
                delta="Outbound target"
            )
        
        with col3:
            st.metric(
                label="Pipeline Value",
                value=f"${advanced.get('total_pipeline_value', 0):,.0f}",
                delta="Total pipeline"
            )
        
        with col4:
            st.metric(
                label="Revenue per Day",
                value=f"${advanced.get('revenue_per_day', 0):,.0f}",
                delta="Daily average"
            )
        
        # LTV and conversion metrics
        col5, col6, col7 = st.columns(3)
        
        with col5:
            st.metric(
                label="LTV Calculation",
                value=f"${baseline.get('ltv_calculation', 0):,.0f}",
                delta=f"{baseline.get('ltv_multiplier', 0)}x multiplier"
            )
        
        with col6:
            st.metric(
                label="Combined Conversion",
                value=f"{advanced.get('combined_conversion_rate', 0):.1f}%",
                delta="Overall rate"
            )
        
        with col7:
            st.metric(
                label="Monthly Run Rate",
                value=f"{advanced.get('monthly_run_rate', 0):,.0f}",
                delta="Applications"
            )
    
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
