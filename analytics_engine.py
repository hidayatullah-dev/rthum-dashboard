"""
Advanced Analytics Engine for Upwork Jobs Dashboard
Implements Excel formulas as Python functions for comprehensive data analysis

WHY: This module centralizes all analytics logic, making it easy to maintain and extend.
The Excel formulas you provided are complex and need to be translated into efficient Python code.

HOW: We use pandas for data manipulation and numpy for calculations, providing the same
functionality as Excel formulas but with better performance and more flexibility.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import streamlit as st
from typing import Dict, List, Tuple, Optional, Any
from config import COLUMNS

class UpworkAnalyticsEngine:
    """
    Advanced analytics engine that implements Excel formulas as Python functions.
    
    WHY: We need a centralized place to handle all the complex analytics from your Excel formulas.
    This class provides a clean interface for all analytics operations.
    
    HOW: Each method corresponds to a specific Excel formula or analysis type,
    using pandas operations to replicate the Excel functionality.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the analytics engine with the job data.
        
        WHY: We need to store the dataframe and prepare it for analysis.
        
        HOW: We store the dataframe and ensure proper data types for analysis.
        """
        self.df = df.copy()
        self._prepare_data()
    
    def _prepare_data(self):
        """
        Prepare the dataframe for analysis by converting data types and cleaning.
        
        WHY: Excel formulas expect specific data types (dates, numbers, etc.).
        We need to ensure our data is in the correct format.
        
        HOW: We convert date columns to datetime, numeric columns to float,
        and clean text columns for consistent analysis.
        """
        # Convert Publish Date to datetime (Column AD)
        if COLUMNS['publish_date_column'] in self.df.columns:
            self.df[COLUMNS['publish_date_column']] = pd.to_datetime(
                self.df[COLUMNS['publish_date_column']], errors='coerce'
            )
        
        # Convert numeric columns
        numeric_columns = [
            COLUMNS['value_column'],  # Score
            COLUMNS['amount_spent_column'],  # Amount spent
            COLUMNS['proposals_column'],  # Proposals
            COLUMNS['interviewing_column'],  # Interviewing
            COLUMNS['invite_sent_column'],  # Invite Sent
            COLUMNS['unanswered_invites_column'],  # Unanswered Invites
            COLUMNS['hourly_rate_column']  # Hourly Rate
        ]
        
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Convert ICP Fit to numeric if it's a score
        if COLUMNS['icp_fit_column'] in self.df.columns:
            self.df[COLUMNS['icp_fit_column']] = pd.to_numeric(
                self.df[COLUMNS['icp_fit_column']], errors='coerce'
            )
    
    def get_weekly_job_counts(self, start_date: str = "2025-07-07") -> pd.DataFrame:
        """
        Calculate weekly job counts based on Publish Date (Column AD).
        Implements Excel formula: =LET(week_num, ROW()-1, start_date, DATE(2025,7,7) + (week_num-1)*7, ...)
        
        WHY: This replicates the Excel formula for weekly job counting.
        It helps track job scraping performance over time.
        
        HOW: We group jobs by week starting from the specified start date,
        counting non-null publish dates for each week.
        """
        if COLUMNS['publish_date_column'] not in self.df.columns:
            st.warning("Publish Date column not found for weekly analysis")
            return pd.DataFrame()
        
        # Filter out null publish dates
        df_clean = self.df.dropna(subset=[COLUMNS['publish_date_column']])
        
        if df_clean.empty:
            return pd.DataFrame()
        
        # Convert start date
        start_dt = pd.to_datetime(start_date)
        
        # Create week ranges
        min_date = df_clean[COLUMNS['publish_date_column']].min()
        max_date = df_clean[COLUMNS['publish_date_column']].max()
        
        # Generate week ranges from start_date
        weeks = []
        current_week_start = start_dt
        
        while current_week_start <= max_date:
            week_end = current_week_start + timedelta(days=6)
            
            # Count jobs in this week
            week_jobs = df_clean[
                (df_clean[COLUMNS['publish_date_column']] >= current_week_start) &
                (df_clean[COLUMNS['publish_date_column']] <= week_end)
            ]
            
            weeks.append({
                'Week': f"Week {len(weeks) + 1}",
                'Week_Start': current_week_start,
                'Week_End': week_end,
                'Job_Count': len(week_jobs),
                'Week_Number': len(weeks) + 1
            })
            
            current_week_start += timedelta(days=7)
        
        return pd.DataFrame(weeks)
    
    def get_daily_job_counts(self) -> pd.DataFrame:
        """
        Calculate daily job counts based on Publish Date.
        Implements Excel formula: =COUNTIFS('Upwork Scraping Version 2'!AD:AD,A2,'Upwork Scraping Version 2'!AD:AD,"<>")
        
        WHY: Daily tracking provides more granular insights than weekly data.
        It helps identify patterns and trends in job posting activity.
        
        HOW: We group jobs by date and count them, similar to Excel's COUNTIFS function.
        """
        if COLUMNS['publish_date_column'] not in self.df.columns:
            st.warning("Publish Date column not found for daily analysis")
            return pd.DataFrame()
        
        # Filter out null publish dates
        df_clean = self.df.dropna(subset=[COLUMNS['publish_date_column']])
        
        if df_clean.empty:
            return pd.DataFrame()
        
        # Group by date and count
        daily_counts = df_clean.groupby(
            df_clean[COLUMNS['publish_date_column']].dt.date
        ).size().reset_index(name='Job_Count')
        
        daily_counts.columns = ['Date', 'Job_Count']
        daily_counts['Date'] = pd.to_datetime(daily_counts['Date'])
        
        return daily_counts.sort_values('Date')
    
    def get_hourly_distribution(self) -> pd.DataFrame:
        """
        Calculate hourly distribution of job postings.
        Implements Excel formula: =SUMPRODUCT(--(HOUR('Upwork Scraping Version 2'!AD:AD)=(ROW()-2)),--('Upwork Scraping Version 2'!AD:AD<>""))
        
        WHY: Understanding when jobs are posted helps optimize application timing.
        This analysis reveals patterns in job posting behavior.
        
        HOW: We extract the hour from publish dates and count jobs by hour.
        """
        if COLUMNS['publish_date_column'] not in self.df.columns:
            st.warning("Publish Date column not found for hourly analysis")
            return pd.DataFrame()
        
        # Filter out null publish dates
        df_clean = self.df.dropna(subset=[COLUMNS['publish_date_column']])
        
        if df_clean.empty:
            return pd.DataFrame()
        
        # Extract hour from publish date
        df_clean['Hour'] = df_clean[COLUMNS['publish_date_column']].dt.hour
        
        # Count jobs by hour
        hourly_counts = df_clean.groupby('Hour').size().reset_index(name='Job_Count')
        
        # Create time labels
        hourly_counts['Time_Label'] = hourly_counts['Hour'].apply(
            lambda x: f"{x:02d}:00"
        )
        
        return hourly_counts.sort_values('Hour')
    
    def get_status_breakdown_weekly(self, start_date: str = "2025-07-07") -> pd.DataFrame:
        """
        Calculate weekly status distribution based on Application Status (Column AE).
        Implements Excel formulas for different status types.
        
        WHY: Tracking application status over time helps measure conversion rates
        and identify trends in job application success.
        
        HOW: We group jobs by week and status, counting each status type per week.
        """
        if COLUMNS['application_status_column'] not in self.df.columns:
            st.warning("Application Status column not found for status analysis")
            return pd.DataFrame()
        
        # Filter out null publish dates and status
        df_clean = self.df.dropna(subset=[
            COLUMNS['publish_date_column'], 
            COLUMNS['application_status_column']
        ])
        
        if df_clean.empty:
            return pd.DataFrame()
        
        # Convert start date
        start_dt = pd.to_datetime(start_date)
        
        # Get unique statuses
        statuses = df_clean[COLUMNS['application_status_column']].unique()
        
        # Create week ranges
        min_date = df_clean[COLUMNS['publish_date_column']].min()
        max_date = df_clean[COLUMNS['publish_date_column']].max()
        
        weeks = []
        current_week_start = start_dt
        
        while current_week_start <= max_date:
            week_end = current_week_start + timedelta(days=6)
            
            # Get jobs in this week
            week_jobs = df_clean[
                (df_clean[COLUMNS['publish_date_column']] >= current_week_start) &
                (df_clean[COLUMNS['publish_date_column']] <= week_end)
            ]
            
            week_data = {
                'Week': f"Week {len(weeks) + 1}",
                'Week_Start': current_week_start,
                'Week_End': week_end,
                'Total_Jobs': len(week_jobs)
            }
            
            # Count each status
            for status in statuses:
                status_count = len(week_jobs[
                    week_jobs[COLUMNS['application_status_column']] == status
                ])
                week_data[f'Status_{status}'] = status_count
            
            weeks.append(week_data)
            current_week_start += timedelta(days=7)
        
        return pd.DataFrame(weeks)
    
    def get_icp_analysis(self) -> Dict[str, Any]:
        """
        Analyze ICP Fit scores and related metrics.
        Implements Excel formulas for ICP analysis.
        
        WHY: ICP (Ideal Customer Profile) analysis helps identify the best job opportunities.
        This provides insights into job quality and targeting effectiveness.
        
        HOW: We calculate various ICP metrics including high-fit jobs, average scores,
        and weekly breakdowns.
        """
        if COLUMNS['icp_fit_column'] not in self.df.columns:
            return {"error": "ICP Fit column not found"}
        
        df_clean = self.df.dropna(subset=[COLUMNS['icp_fit_column']])
        
        if df_clean.empty:
            return {"error": "No ICP data available"}
        
        # High ICP fit jobs (assuming scale 1-5, where 5 is highest)
        high_icp_jobs = len(df_clean[df_clean[COLUMNS['icp_fit_column']] >= 4])
        
        # Average ICP fit score
        avg_icp_score = df_clean[COLUMNS['icp_fit_column']].mean()
        
        # ICP distribution
        icp_distribution = df_clean[COLUMNS['icp_fit_column']].value_counts().sort_index()
        
        return {
            "high_icp_jobs": high_icp_jobs,
            "avg_icp_score": avg_icp_score,
            "total_icp_jobs": len(df_clean),
            "icp_distribution": icp_distribution.to_dict(),
            "high_icp_percentage": (high_icp_jobs / len(df_clean)) * 100
        }
    
    def get_experience_level_distribution(self) -> pd.DataFrame:
        """
        Analyze experience level distribution.
        Implements Excel formulas for experience level analysis.
        
        WHY: Understanding the experience level of jobs helps target the right opportunities.
        This analysis reveals market trends and job requirements.
        
        HOW: We count jobs by experience level and calculate percentages.
        """
        if COLUMNS['experience_level_column'] not in self.df.columns:
            st.warning("Experience Level column not found")
            return pd.DataFrame()
        
        df_clean = self.df.dropna(subset=[COLUMNS['experience_level_column']])
        
        if df_clean.empty:
            return pd.DataFrame()
        
        # Count by experience level
        exp_distribution = df_clean[COLUMNS['experience_level_column']].value_counts().reset_index()
        exp_distribution.columns = ['Experience_Level', 'Count']
        
        # Calculate percentage
        exp_distribution['Percentage'] = (exp_distribution['Count'] / exp_distribution['Count'].sum()) * 100
        
        return exp_distribution.sort_values('Count', ascending=False)
    
    def get_hourly_rate_analysis(self) -> Dict[str, Any]:
        """
        Analyze hourly rates and rate distributions.
        Implements Excel formulas for hourly rate analysis.
        
        WHY: Rate analysis helps understand market pricing and identify high-value opportunities.
        This provides insights into budget trends and job value.
        
        HOW: We calculate average rates, count jobs in different rate ranges,
        and identify high-paying opportunities.
        """
        if COLUMNS['hourly_rate_column'] not in self.df.columns:
            return {"error": "Hourly Rate column not found"}
        
        df_clean = self.df.dropna(subset=[COLUMNS['hourly_rate_column']])
        
        if df_clean.empty:
            return {"error": "No hourly rate data available"}
        
        # Average hourly rate
        avg_rate = df_clean[COLUMNS['hourly_rate_column']].mean()
        
        # High paying jobs (>$50/hour)
        high_paying = len(df_clean[df_clean[COLUMNS['hourly_rate_column']] > 50])
        
        # Rate ranges
        rate_ranges = {
            "0-25": len(df_clean[(df_clean[COLUMNS['hourly_rate_column']] >= 0) & 
                                (df_clean[COLUMNS['hourly_rate_column']] <= 25)]),
            "26-50": len(df_clean[(df_clean[COLUMNS['hourly_rate_column']] > 25) & 
                                (df_clean[COLUMNS['hourly_rate_column']] <= 50)]),
            "51-100": len(df_clean[(df_clean[COLUMNS['hourly_rate_column']] > 50) & 
                                 (df_clean[COLUMNS['hourly_rate_column']] <= 100)]),
            "100+": len(df_clean[df_clean[COLUMNS['hourly_rate_column']] > 100])
        }
        
        return {
            "avg_hourly_rate": avg_rate,
            "high_paying_jobs": high_paying,
            "total_jobs": len(df_clean),
            "rate_ranges": rate_ranges,
            "high_paying_percentage": (high_paying / len(df_clean)) * 100
        }
    
    def get_country_analysis(self, top_n: int = 10) -> pd.DataFrame:
        """
        Analyze job distribution by country.
        Implements Excel formulas for country analysis.
        
        WHY: Geographic analysis helps understand market distribution and target regions.
        This provides insights into global job posting patterns.
        
        HOW: We count jobs by country and identify the top markets.
        """
        if COLUMNS['country_column'] not in self.df.columns:
            st.warning("Country column not found")
            return pd.DataFrame()
        
        df_clean = self.df.dropna(subset=[COLUMNS['country_column']])
        
        if df_clean.empty:
            return pd.DataFrame()
        
        # Count by country
        country_counts = df_clean[COLUMNS['country_column']].value_counts().head(top_n).reset_index()
        country_counts.columns = ['Country', 'Job_Count']
        
        # Calculate percentage
        country_counts['Percentage'] = (country_counts['Job_Count'] / country_counts['Job_Count'].sum()) * 100
        
        return country_counts
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """
        Calculate real-time dashboard metrics.
        Implements Excel formulas for summary statistics and performance tracking.
        
        WHY: Real-time metrics provide immediate insights into current performance.
        This helps track progress and identify issues quickly.
        
        HOW: We calculate various KPIs including total jobs, application rates,
        and performance indicators.
        """
        # Total jobs scraped
        total_jobs = len(self.df)
        
        # Jobs scraped today
        today = datetime.now().date()
        today_jobs = 0
        if COLUMNS['publish_date_column'] in self.df.columns:
            today_jobs = len(self.df[
                self.df[COLUMNS['publish_date_column']].dt.date == today
            ])
        
        # Jobs scraped this week
        week_start = today - timedelta(days=today.weekday())
        week_jobs = 0
        if COLUMNS['publish_date_column'] in self.df.columns:
            week_jobs = len(self.df[
                self.df[COLUMNS['publish_date_column']].dt.date >= week_start
            ])
        
        # Jobs scraped this month
        month_start = today.replace(day=1)
        month_jobs = 0
        if COLUMNS['publish_date_column'] in self.df.columns:
            month_jobs = len(self.df[
                self.df[COLUMNS['publish_date_column']].dt.date >= month_start
            ])
        
        # Application rate
        applied_jobs = 0
        if COLUMNS['application_status_column'] in self.df.columns:
            applied_jobs = len(self.df[
                self.df[COLUMNS['application_status_column']] == 'Applied'
            ])
        
        application_rate = (applied_jobs / total_jobs * 100) if total_jobs > 0 else 0
        
        # Average daily scraping rate
        if COLUMNS['publish_date_column'] in self.df.columns:
            first_date = self.df[COLUMNS['publish_date_column']].min()
            last_date = self.df[COLUMNS['publish_date_column']].max()
            if pd.notna(first_date) and pd.notna(last_date):
                days_active = (last_date - first_date).days + 1
                avg_daily_rate = total_jobs / days_active if days_active > 0 else 0
            else:
                avg_daily_rate = 0
        else:
            avg_daily_rate = 0
        
        # High-value opportunities (High ICP + Good Rate)
        high_value_jobs = 0
        if (COLUMNS['icp_fit_column'] in self.df.columns and 
            COLUMNS['hourly_rate_column'] in self.df.columns and
            COLUMNS['application_status_column'] in self.df.columns):
            
            high_value_jobs = len(self.df[
                (self.df[COLUMNS['icp_fit_column']] >= 4) &
                (self.df[COLUMNS['hourly_rate_column']] > 30) &
                (self.df[COLUMNS['application_status_column']] != 'Not interested')
            ])
        
        return {
            "total_jobs": total_jobs,
            "today_jobs": today_jobs,
            "week_jobs": week_jobs,
            "month_jobs": month_jobs,
            "applied_jobs": applied_jobs,
            "application_rate": application_rate,
            "avg_daily_rate": avg_daily_rate,
            "high_value_jobs": high_value_jobs,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def get_conversion_funnel(self) -> Dict[str, int]:
        """
        Calculate conversion funnel metrics.
        Implements Excel formulas for funnel analysis.
        
        WHY: Conversion funnel analysis helps track the job application process
        from initial scraping to final application.
        
        HOW: We count jobs at each stage of the funnel process.
        """
        # Total scraped
        total_scraped = len(self.df)
        
        # Passed filters (assuming this is tracked in a column)
        passed_filters = 0
        if 'Passed_filters' in self.df.columns:
            passed_filters = len(self.df[self.df['Passed_filters'] == 'Yes'])
        
        # Applied
        applied = 0
        if COLUMNS['application_status_column'] in self.df.columns:
            applied = len(self.df[
                self.df[COLUMNS['application_status_column']] == 'Applied'
            ])
        
        return {
            "total_scraped": total_scraped,
            "passed_filters": passed_filters,
            "applied": applied,
            "filter_pass_rate": (passed_filters / total_scraped * 100) if total_scraped > 0 else 0,
            "application_rate": (applied / total_scraped * 100) if total_scraped > 0 else 0
        }
