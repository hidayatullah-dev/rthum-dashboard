"""
Proposal Tracking Analytics Engine
Implements Google Sheets formulas for proposal tracking dashboard

WHY: This module centralizes all proposal tracking analytics, translating your Google Sheets
formulas into efficient Python functions for real-time dashboard updates.

HOW: We use pandas for data manipulation and implement each Excel formula as a Python function,
ensuring accurate calculations and real-time data processing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import streamlit as st
from config import PROPOSAL_COLUMNS, PROPOSAL_TRACKING_SHEET, PROPOSAL_WORKSHEET

class ProposalTrackingEngine:
    """
    Advanced proposal tracking analytics engine that implements Google Sheets formulas.
    
    WHY: We need a centralized place to handle all proposal tracking calculations from your
    Google Sheets formulas. This class provides real-time analytics for your proposal funnel.
    
    HOW: Each method corresponds to a specific Google Sheets formula, using pandas operations
    to replicate the Excel functionality with better performance and real-time updates.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the proposal tracking engine with proposal data.
        
        WHY: We need to store the dataframe and prepare it for analysis.
        
        HOW: We store the dataframe and ensure proper data types for calculations.
        """
        self.df = df.copy()
        self._prepare_data()
    
    def _prepare_data(self):
        """
        Prepare the dataframe for analysis by converting data types and cleaning.
        
        WHY: Excel formulas expect specific data types (numbers, percentages, etc.).
        We need to ensure our data is in the correct format for calculations.
        
        HOW: We convert numeric columns to float and clean text columns for consistent analysis.
        """
        # Convert all numeric columns
        numeric_columns = [
            PROPOSAL_COLUMNS['applications_sent'],
            PROPOSAL_COLUMNS['replies'],
            PROPOSAL_COLUMNS['interviews'],
            PROPOSAL_COLUMNS['jobs_won'],
            PROPOSAL_COLUMNS['average_deal_size'],
            PROPOSAL_COLUMNS['ltv_multiplier'],
            PROPOSAL_COLUMNS['monthly_gross_revenue'],
            PROPOSAL_COLUMNS['target_applications'],
            PROPOSAL_COLUMNS['target_replies'],
            PROPOSAL_COLUMNS['target_interviews'],
            PROPOSAL_COLUMNS['target_jobs_won'],
            PROPOSAL_COLUMNS['target_deal_size'],
            PROPOSAL_COLUMNS['target_ltv'],
            PROPOSAL_COLUMNS['target_revenue'],
            PROPOSAL_COLUMNS['impressions'],
            PROPOSAL_COLUMNS['profile_views'],
            PROPOSAL_COLUMNS['invites'],
            PROPOSAL_COLUMNS['inbound_hires'],
            PROPOSAL_COLUMNS['inbound_deal_size'],
            PROPOSAL_COLUMNS['inbound_ltv'],
            PROPOSAL_COLUMNS['inbound_revenue'],
            PROPOSAL_COLUMNS['daily_applications'],
            PROPOSAL_COLUMNS['daily_replies'],
            PROPOSAL_COLUMNS['daily_interviews'],
            PROPOSAL_COLUMNS['daily_jobs_won'],
            PROPOSAL_COLUMNS['daily_impressions'],
            PROPOSAL_COLUMNS['daily_profile_views'],
            PROPOSAL_COLUMNS['weekly_invites'],
            PROPOSAL_COLUMNS['monthly_hires']
        ]
        
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
    
    def get_baseline_2025_data(self) -> Dict[str, Any]:
        """
        Get baseline 2025 data metrics.
        Implements Excel formulas for core metrics (columns B4-B7, B10-B12).
        
        WHY: This provides the foundation metrics for all other calculations.
        It represents your actual performance data for the current period.
        
        HOW: We extract the core metrics and calculate derived values using
        the same logic as your Excel formulas.
        """
        # Core metrics (assuming these are in the first row of data)
        applications_sent = self._get_value(PROPOSAL_COLUMNS['applications_sent'], 0)
        replies = self._get_value(PROPOSAL_COLUMNS['replies'], 0)
        interviews = self._get_value(PROPOSAL_COLUMNS['interviews'], 0)
        jobs_won = self._get_value(PROPOSAL_COLUMNS['jobs_won'], 0)
        average_deal_size = self._get_value(PROPOSAL_COLUMNS['average_deal_size'], 4500)  # Default 4500
        ltv_multiplier = self._get_value(PROPOSAL_COLUMNS['ltv_multiplier'], 60)  # Default 60
        
        # Conversion rate calculations (Excel formulas C5, C6, C7)
        reply_conversion_rate = self._safe_divide(replies, applications_sent) * 100
        interview_conversion_rate = self._safe_divide(interviews, replies) * 100
        job_won_conversion_rate = self._safe_divide(jobs_won, interviews) * 100
        overall_hire_rate = self._safe_divide(jobs_won, applications_sent) * 100
        
        # Financial metrics (Excel formulas B11, B12)
        ltv_calculation = jobs_won * average_deal_size * ltv_multiplier
        monthly_gross_revenue = jobs_won * average_deal_size
        
        return {
            "applications_sent": applications_sent,
            "replies": replies,
            "interviews": interviews,
            "jobs_won": jobs_won,
            "average_deal_size": average_deal_size,
            "ltv_multiplier": ltv_multiplier,
            "reply_conversion_rate": reply_conversion_rate,
            "interview_conversion_rate": interview_conversion_rate,
            "job_won_conversion_rate": job_won_conversion_rate,
            "overall_hire_rate": overall_hire_rate,
            "ltv_calculation": ltv_calculation,
            "monthly_gross_revenue": monthly_gross_revenue
        }
    
    def get_outbound_monthly_goals(self) -> Dict[str, Any]:
        """
        Get outbound monthly goals (90% section).
        Implements Excel formulas for target data (columns E4-E7, E10-E12, E14-E16).
        
        WHY: This shows your target performance for outbound activities.
        It helps track progress against goals and identify gaps.
        
        HOW: We calculate target metrics based on conversion rates and
        project financial outcomes using your Excel formulas.
        """
        # Target data (assuming these are in the first row of data)
        target_applications = self._get_value(PROPOSAL_COLUMNS['target_applications'], 197)
        target_deal_size = self._get_value(PROPOSAL_COLUMNS['target_deal_size'], 4500)
        
        # Target calculations based on conversion rates (Excel formulas E5, E6, E7)
        target_replies = round(target_applications * 0.24)  # 24% conversion rate
        target_interviews = round(target_replies * 0.66)    # 66% conversion rate
        target_jobs_won = round(target_interviews * 0.26)   # 26% conversion rate
        
        # Conversion rate display (Excel formulas F5, F6, F7)
        reply_conversion_rate = 24
        interview_conversion_rate = 66
        job_won_conversion_rate = 26
        overall_hire_rate = self._safe_divide(target_jobs_won, target_applications) * 100
        
        # Financial projections (Excel formulas E11, E12, E14, E15, E16)
        ltv_projection = target_jobs_won * target_deal_size * 96  # 96x multiplier
        monthly_gross_revenue = target_jobs_won * target_deal_size
        
        # Combined outbound + inbound (we'll get inbound from separate method)
        inbound_revenue = self._get_value(PROPOSAL_COLUMNS['inbound_revenue'], 0)
        combined_revenue = monthly_gross_revenue + inbound_revenue
        commission = combined_revenue * 0.1  # 10% commission
        net_revenue = combined_revenue * 0.9  # 90% net revenue
        
        return {
            "target_applications": target_applications,
            "target_replies": target_replies,
            "target_interviews": target_interviews,
            "target_jobs_won": target_jobs_won,
            "target_deal_size": target_deal_size,
            "reply_conversion_rate": reply_conversion_rate,
            "interview_conversion_rate": interview_conversion_rate,
            "job_won_conversion_rate": job_won_conversion_rate,
            "overall_hire_rate": overall_hire_rate,
            "ltv_projection": ltv_projection,
            "monthly_gross_revenue": monthly_gross_revenue,
            "combined_revenue": combined_revenue,
            "commission": commission,
            "net_revenue": net_revenue
        }
    
    def get_inbound_monthly_goals(self) -> Dict[str, Any]:
        """
        Get inbound monthly goals (90% section).
        Implements Excel formulas for inbound metrics (columns H4-H7, H9-H12, H15-H16).
        
        WHY: This tracks your inbound marketing performance and goals.
        It shows how well your profile and visibility are performing.
        
        HOW: We calculate inbound metrics based on conversion funnels
        and project financial outcomes using your Excel formulas.
        """
        # Target metrics (assuming these are in the first row of data)
        target_impressions = self._get_value(PROPOSAL_COLUMNS['impressions'], 1700)
        inbound_deal_size = self._get_value(PROPOSAL_COLUMNS['inbound_deal_size'], 4500)
        
        # Target calculations based on conversion rates (Excel formulas H5, H6, H7)
        target_profile_views = round(target_impressions * 0.03)  # 3% of impressions
        target_invites = round(target_profile_views * 0.10)      # 10% of profile views
        target_hires = round(target_invites * 0.20)             # 20% of invites
        
        # Conversion rates (Excel formulas I5, I6, I7)
        profile_view_conversion_rate = self._safe_divide(target_profile_views, target_impressions) * 100
        invite_conversion_rate = self._safe_divide(target_invites, target_profile_views) * 100
        hire_conversion_rate = self._safe_divide(target_hires, target_invites) * 100
        
        # Overachievement targets (Excel formulas K5, K6, K7)
        overachieve_profile_views = round(target_impressions * 0.05)  # 5% of impressions
        overachieve_invites = round(overachieve_profile_views * 0.40)  # 40% of profile views
        overachieve_hires = round(overachieve_invites * 0.20)         # 20% of invites
        
        # Financial metrics (Excel formulas H9, H11, H12)
        hire_rate = self._safe_divide(target_hires, target_invites) * 100
        ltv = target_hires * inbound_deal_size * 12  # 12x multiplier
        monthly_gross_revenue = target_hires * inbound_deal_size
        
        # Daily breakdown (Excel formulas H15, H16)
        impressions_per_day = round(target_impressions / 30)
        impressions_per_week = round(target_impressions / 4.33)
        
        return {
            "target_impressions": target_impressions,
            "target_profile_views": target_profile_views,
            "target_invites": target_invites,
            "target_hires": target_hires,
            "inbound_deal_size": inbound_deal_size,
            "profile_view_conversion_rate": profile_view_conversion_rate,
            "invite_conversion_rate": invite_conversion_rate,
            "hire_conversion_rate": hire_conversion_rate,
            "overachieve_profile_views": overachieve_profile_views,
            "overachieve_invites": overachieve_invites,
            "overachieve_hires": overachieve_hires,
            "hire_rate": hire_rate,
            "ltv": ltv,
            "monthly_gross_revenue": monthly_gross_revenue,
            "impressions_per_day": impressions_per_day,
            "impressions_per_week": impressions_per_week
        }
    
    def get_daily_activity_targets(self) -> Dict[str, Any]:
        """
        Get daily activity targets.
        Implements Excel formulas for daily targets (columns M4-M7, M10-M13).
        
        WHY: This breaks down monthly goals into daily actionable targets.
        It helps with daily planning and progress tracking.
        
        HOW: We divide monthly targets by 30 days and calculate
        weekly targets using your Excel formulas.
        """
        # Get monthly targets from other methods
        outbound_goals = self.get_outbound_monthly_goals()
        inbound_goals = self.get_inbound_monthly_goals()
        
        # Outbound daily targets (Excel formulas M4, M5, M6, M7)
        daily_applications = round(outbound_goals['target_applications'] / 30)
        daily_replies = round(outbound_goals['target_replies'] / 30)
        daily_interviews = round(outbound_goals['target_interviews'] / 30)
        daily_jobs_won = round(outbound_goals['target_jobs_won'] / 30, 1)
        
        # Inbound daily targets (Excel formulas M10, M11, M12, M13)
        daily_impressions = round(inbound_goals['target_impressions'] / 30)
        daily_profile_views = round(inbound_goals['target_profile_views'] / 30)
        weekly_invites = round(inbound_goals['target_invites'] / 4.33)
        monthly_hires = inbound_goals['target_hires']
        
        return {
            "daily_applications": daily_applications,
            "daily_replies": daily_replies,
            "daily_interviews": daily_interviews,
            "daily_jobs_won": daily_jobs_won,
            "daily_impressions": daily_impressions,
            "daily_profile_views": daily_profile_views,
            "weekly_invites": weekly_invites,
            "monthly_hires": monthly_hires
        }
    
    def get_performance_tracking(self) -> Dict[str, Any]:
        """
        Get performance tracking metrics (Actual vs Target).
        Implements Excel formulas for performance tracking.
        
        WHY: This shows how well you're performing against your targets.
        It helps identify areas that need attention or improvement.
        
        HOW: We compare actual performance against targets and calculate
        performance percentages using your Excel formulas.
        """
        baseline_data = self.get_baseline_2025_data()
        outbound_goals = self.get_outbound_monthly_goals()
        inbound_goals = self.get_inbound_monthly_goals()
        
        # Outbound performance percentage
        outbound_performance = self._safe_divide(
            baseline_data['applications_sent'], 
            outbound_goals['target_applications']
        ) * 100
        
        # Inbound performance percentage (assuming we have actual impressions)
        actual_impressions = self._get_value(PROPOSAL_COLUMNS['impressions'], 0)
        inbound_performance = self._safe_divide(
            actual_impressions, 
            inbound_goals['target_impressions']
        ) * 100
        
        # Revenue performance
        actual_revenue = baseline_data['monthly_gross_revenue']
        target_revenue = outbound_goals['monthly_gross_revenue']
        revenue_performance = self._safe_divide(actual_revenue, target_revenue) * 100
        
        return {
            "outbound_performance": outbound_performance,
            "inbound_performance": inbound_performance,
            "revenue_performance": revenue_performance,
            "actual_applications": baseline_data['applications_sent'],
            "target_applications": outbound_goals['target_applications'],
            "actual_impressions": actual_impressions,
            "target_impressions": inbound_goals['target_impressions'],
            "actual_revenue": actual_revenue,
            "target_revenue": target_revenue
        }
    
    def get_advanced_analytics(self) -> Dict[str, Any]:
        """
        Get advanced analytics including trend analysis and forecasting.
        Implements Excel formulas for trend analysis and forecasting.
        
        WHY: This provides deeper insights into performance trends and future projections.
        It helps with strategic planning and goal setting.
        
        HOW: We calculate growth rates, run rates, and projections
        using your Excel formulas for trend analysis.
        """
        baseline_data = self.get_baseline_2025_data()
        outbound_goals = self.get_outbound_monthly_goals()
        inbound_goals = self.get_inbound_monthly_goals()
        
        # Total pipeline value (Excel formula: B7*B10 + E7*E10 + H7*H10)
        total_pipeline_value = (
            baseline_data['jobs_won'] * baseline_data['average_deal_size'] +
            outbound_goals['target_jobs_won'] * outbound_goals['target_deal_size'] +
            inbound_goals['target_hires'] * inbound_goals['inbound_deal_size']
        )
        
        # Combined conversion rate (Excel formula: (B7+E7+H7)/(B4+E4+H4)*100)
        total_applications = (
            baseline_data['applications_sent'] + 
            outbound_goals['target_applications'] + 
            inbound_goals['target_impressions']  # Using impressions as proxy for applications
        )
        total_hires = (
            baseline_data['jobs_won'] + 
            outbound_goals['target_jobs_won'] + 
            inbound_goals['target_hires']
        )
        combined_conversion_rate = self._safe_divide(total_hires, total_applications) * 100
        
        # Revenue per day (Excel formula: (B12+E12+H12)/30)
        total_revenue = (
            baseline_data['monthly_gross_revenue'] + 
            outbound_goals['monthly_gross_revenue'] + 
            inbound_goals['monthly_gross_revenue']
        )
        revenue_per_day = total_revenue / 30
        
        # Monthly run rate (Excel formula: CURRENT_MONTH_TOTAL/DAY(TODAY())*30)
        current_day = datetime.now().day
        if current_day > 0:
            monthly_run_rate = (baseline_data['applications_sent'] / current_day) * 30
        else:
            monthly_run_rate = 0
        
        # End of month projection (Excel formula: CURRENT_MONTH_TOTAL/DAY(TODAY())*DAY(EOMONTH(TODAY(),0)))
        if current_day > 0:
            end_of_month = datetime.now().replace(day=1) + timedelta(days=32)
            end_of_month = end_of_month.replace(day=1) - timedelta(days=1)
            end_of_month_projection = (baseline_data['applications_sent'] / current_day) * end_of_month.day
        else:
            end_of_month_projection = 0
        
        # Quarter projection (Excel formula: CURRENT_MONTH_TOTAL*3)
        quarter_projection = baseline_data['applications_sent'] * 3
        
        return {
            "total_pipeline_value": total_pipeline_value,
            "combined_conversion_rate": combined_conversion_rate,
            "revenue_per_day": revenue_per_day,
            "monthly_run_rate": monthly_run_rate,
            "end_of_month_projection": end_of_month_projection,
            "quarter_projection": quarter_projection,
            "total_applications": total_applications,
            "total_hires": total_hires,
            "total_revenue": total_revenue
        }
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """
        Get dashboard summary with key metrics.
        Implements Excel formulas for dashboard summary.
        
        WHY: This provides a high-level overview of all key performance indicators.
        It's perfect for executive dashboards and quick status updates.
        
        HOW: We combine all metrics into a comprehensive summary
        using your Excel formulas for key metrics.
        """
        baseline_data = self.get_baseline_2025_data()
        outbound_goals = self.get_outbound_monthly_goals()
        inbound_goals = self.get_inbound_monthly_goals()
        performance = self.get_performance_tracking()
        advanced = self.get_advanced_analytics()
        
        return {
            "baseline": baseline_data,
            "outbound_goals": outbound_goals,
            "inbound_goals": inbound_goals,
            "performance": performance,
            "advanced": advanced,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _get_value(self, column_name: str, default_value: Any = 0) -> Any:
        """
        Get a value from the dataframe with error handling.
        
        WHY: We need safe access to dataframe values with fallbacks.
        
        HOW: We check if the column exists and has data, returning a default if not.
        """
        if column_name in self.df.columns and not self.df[column_name].isna().all():
            return self.df[column_name].iloc[0] if len(self.df) > 0 else default_value
        return default_value
    
    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """
        Safe division that handles zero denominators.
        
        WHY: Excel formulas handle division by zero gracefully.
        We need the same behavior in Python.
        
        HOW: We return 0 if the denominator is zero, otherwise perform normal division.
        """
        if denominator == 0:
            return 0
        return numerator / denominator
