"""
Advanced Visualization Module for Churn Analysis
Professional, publication-ready charts and dashboards
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

class AdvancedVisualizer:
    def __init__(self, df):
        self.df = df
        self.style_config = {
            'color_palette': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B'],
            'style': 'seaborn-v0_8-whitegrid',
            'figsize': (12, 8),
            'dpi': 300
        }
        plt.style.use(self.style_config['style'])
        
    def create_comprehensive_dashboard(self, save_path='visualizations/'):
        """Create a comprehensive churn analysis dashboard"""
        print("📊 Creating comprehensive analysis dashboard...")
        
        # Create subplot structure
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3)
        
        # 1. Churn Distribution (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_churn_distribution(ax1)
        
        # 2. Feature Importance (Top Middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_feature_importance(ax2)
        
        # 3. Correlation Heatmap (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_correlation_heatmap(ax3)
        
        # 4. Tenure Analysis (Middle Left)
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_tenure_analysis(ax4)
        
        # 5. Price Sensitivity Analysis (Middle)
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_price_sensitivity(ax5)
        
        # 6. Usage Patterns (Middle Right)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_usage_patterns(ax6)
        
        # 7. Customer Segments (Bottom)
        ax7 = fig.add_subplot(gs[2, :])
        self._plot_customer_segments(ax7)
        
        plt.suptitle('GlobalStream Inc. - Comprehensive Churn Analysis Dashboard\nData-Driven Insights for Customer Retention', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f'{save_path}/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comprehensive dashboard saved to {save_path}")
    
    def _plot_churn_distribution(self, ax):
        """Plot churn distribution with enhanced styling"""
        churn_counts = self.df['churn'].value_counts()
        colors = [self.style_config['color_palette'][0], self.style_config['color_palette'][1]]
        
        wedges, texts, autotexts = ax.pie(churn_counts, 
                                         labels=['Retained', 'Churned'],
                                         autopct='%1.1f%%',
                                         colors=colors,
                                         startangle=90,
                                         explode=(0.05, 0))
        
        # Enhance text styling
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax.set_title('Customer Churn Distribution\nOverall Business Impact', 
                    fontweight='bold', fontsize=12, pad=20)
    
    def _plot_feature_importance(self, ax):
        """Plot feature importance for churn prediction"""
        # Calculate correlations with churn
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        correlations = []
        
        for col in numerical_cols:
            if col != 'churn':
                corr = self.df[col].corr(self.df['churn'])
                correlations.append((col, corr))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        features, corr_values = zip(*correlations[:8])
        
        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        colors = [self.style_config['color_palette'][1] if x > 0 else self.style_config['color_palette'][0] for x in corr_values]
        
        bars = ax.barh(y_pos, corr_values, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f.replace('_', ' ').title() for f in features], fontsize=9)
        ax.set_xlabel('Correlation with Churn')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value annotations
        for i, (bar, value) in enumerate(zip(bars, corr_values)):
            ax.text(value + (0.01 if value >= 0 else -0.01), i, f'{value:.3f}', 
                   ha='left' if value >= 0 else 'right', va='center', fontsize=8, fontweight='bold')
        
        ax.set_title('Top Churn Drivers\nFeature Correlation Analysis', 
                    fontweight='bold', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
    
    def _plot_correlation_heatmap(self, ax):
        """Plot correlation heatmap for key features"""
        # Select top correlated features
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_with_churn = self.df[numerical_cols].corr()['churn'].abs().sort_values(ascending=False)
        top_features = corr_with_churn.index[:8].tolist()
        
        # Create correlation matrix
        corr_matrix = self.df[top_features].corr()
        
        # Plot heatmap
        im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(top_features)))
        ax.set_yticks(np.arange(len(top_features)))
        ax.set_xticklabels([f.replace('_', '\n').title() for f in top_features], fontsize=8, rotation=45)
        ax.set_yticklabels([f.replace('_', '\n').title() for f in top_features], fontsize=8)
        
        # Add correlation values
        for i in range(len(top_features)):
            for j in range(len(top_features)):
                ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                       ha='center', va='center', fontsize=7, 
                       color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
        
        ax.set_title('Feature Correlation Matrix\nRelationship Overview', 
                    fontweight='bold', fontsize=12, pad=20)
        plt.colorbar(im, ax=ax, shrink=0.6)
    
    def _plot_tenure_analysis(self, ax):
        """Plot tenure analysis with churn rates"""
        if 'tenure_days' not in self.df.columns:
            return
        
        # Create tenure bins
        self.df['tenure_bin'] = pd.cut(self.df['tenure_days'], bins=6)
        tenure_churn = self.df.groupby('tenure_bin')['churn'].mean()
        
        # Plot
        bars = ax.bar(range(len(tenure_churn)), tenure_churn.values * 100, 
                     color=self.style_config['color_palette'][1], alpha=0.7)
        
        ax.set_xlabel('Tenure (Days)')
        ax.set_ylabel('Churn Rate (%)')
        ax.set_xticks(range(len(tenure_churn)))
        ax.set_xticklabels([str(x) for x in tenure_churn.index], rotation=45, fontsize=8)
        ax.set_title('Churn Rate by Customer Tenure\nEarly-Life Attrition Pattern', 
                    fontweight='bold', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, tenure_churn.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   f'{value*100:.1f}%', ha='center', va='bottom', fontsize=8)
    
    def _plot_price_sensitivity(self, ax):
        """Plot price sensitivity analysis"""
        if not all(col in self.df.columns for col in ['monthly_charges', 'daily_viewing_hours', 'churn']):
            return
        
        # Create scatter plot
        churned = self.df[self.df['churn'] == 1]
        retained = self.df[self.df['churn'] == 0]
        
        ax.scatter(retained['monthly_charges'], retained['daily_viewing_hours'], 
                  alpha=0.6, color=self.style_config['color_palette'][0], 
                  label='Retained', s=30)
        ax.scatter(churned['monthly_charges'], churned['daily_viewing_hours'], 
                  alpha=0.8, color=self.style_config['color_palette'][1], 
                  label='Churned', s=40, marker='x')
        
        ax.set_xlabel('Monthly Charges ($)')
        ax.set_ylabel('Daily Viewing Hours')
        ax.legend(fontsize=9)
        ax.set_title('Price vs Usage: Churn Patterns\nIdentifying Price-Sensitive Customers', 
                    fontweight='bold', fontsize=12)
        ax.grid(alpha=0.3)
    
    def _plot_usage_patterns(self, ax):
        """Plot usage pattern analysis"""
        if 'daily_viewing_hours' not in self.df.columns:
            return
        
        # Create density plot
        churned_usage = self.df[self.df['churn'] == 1]['daily_viewing_hours']
        retained_usage = self.df[self.df['churn'] == 0]['daily_viewing_hours']
        
        churned_usage.plot.density(ax=ax, color=self.style_config['color_palette'][1], 
                                  label='Churned', linewidth=2)
        retained_usage.plot.density(ax=ax, color=self.style_config['color_palette'][0], 
                                   label='Retained', linewidth=2)
        
        ax.set_xlabel('Daily Viewing Hours')
        ax.set_ylabel('Density')
        ax.legend(fontsize=9)
        ax.set_title('Usage Pattern Distribution\nEngagement Level Comparison', 
                    fontweight='bold', fontsize=12)
        ax.grid(alpha=0.3)
        ax.set_xlim(0)
    
    def _plot_customer_segments(self, ax):
        """Plot customer segmentation analysis"""
        if not all(col in self.df.columns for col in ['monthly_charges', 'daily_viewing_hours', 'tenure_days']):
            return
        
        # Create customer segments
        conditions = [
            (self.df['monthly_charges'] > self.df['monthly_charges'].median()) & 
            (self.df['daily_viewing_hours'] < self.df['daily_viewing_hours'].median()),
            
            (self.df['tenure_days'] > self.df['tenure_days'].median()) & 
            (self.df['daily_viewing_hours'] > self.df['daily_viewing_hours'].median()),
            
            (self.df['monthly_charges'] < self.df['monthly_charges'].median()) & 
            (self.df['tenure_days'] > self.df['tenure_days'].median())
        ]
        
        segments = ['High Risk', 'Loyal', 'Value']
        self.df['segment'] = np.select(conditions, segments, default='Other')
        
        # Calculate segment sizes and churn rates
        segment_summary = self.df.groupby('segment').agg({
            'churn': ['count', 'mean']
        }).round(3)
        
        segment_summary.columns = ['count', 'churn_rate']
        segment_summary = segment_summary.loc[segments + ['Other']]
        
        # Create grouped bar plot
        x = np.arange(len(segment_summary))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, segment_summary['count'], width, 
                      label='Customer Count', color=self.style_config['color_palette'][0], alpha=0.7)
        bars2 = ax.bar(x + width/2, segment_summary['churn_rate'] * 100, width, 
                      label='Churn Rate (%)', color=self.style_config['color_palette'][1], alpha=0.7)
        
        ax.set_xlabel('Customer Segment')
        ax.set_ylabel('Metrics')
        ax.set_title('Customer Segmentation Analysis\nStrategic Group Identification', 
                    fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(segment_summary.index, rotation=0)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + (0.01 * max(segment_summary['count'])), 
                       f'{height:.0f}' if bars == bars1 else f'{height:.1f}%', 
                       ha='center', va='bottom', fontsize=8)
    
    def create_interactive_dashboard(self):
        """Create interactive Plotly dashboard"""
        print("🎨 Creating interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Churn Distribution', 'Top Churn Drivers', 'Customer Segments',
                'Price vs Usage Analysis', 'Tenure Impact', 'Usage Patterns'
            ),
            specs=[
                [{"type": "pie"}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "histogram"}]
            ]
        )
        
        # 1. Churn Distribution (Pie chart)
        churn_counts = self.df['churn'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=['Retained', 'Churned'],
                values=churn_counts.values,
                marker_colors=['#2E86AB', '#A23B72'],
                textinfo='percent+label',
                hole=0.3
            ),
            row=1, col=1
        )
        
        # 2. Feature Importance (Bar chart)
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        correlations = []
        for col in numerical_cols:
            if col != 'churn':
                corr = self.df[col].corr(self.df['churn'])
                correlations.append((col, corr))
        
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        features, corr_values = zip(*correlations[:6])
        
        fig.add_trace(
            go.Bar(
                x=corr_values,
                y=[f.replace('_', ' ').title() for f in features],
                orientation='h',
                marker_color=['#A23B72' if x > 0 else '#2E86AB' for x in corr_values]
            ),
            row=1, col=2
        )
        
        # 3. Update layout
        fig.update_layout(
            title_text="GlobalStream Inc. - Interactive Churn Analysis Dashboard",
            height=800,
            showlegend=False,
            template="plotly_white"
        )
        
        # Save interactive dashboard
        fig.write_html("visualizations/interactive_dashboard.html")
        print("✅ Interactive dashboard saved as HTML")
    
    def create_executive_summary_chart(self, save_path='visualizations/'):
        """Create executive summary chart for presentations"""
        print("📈 Creating executive summary chart...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Key Metric Summary
        metrics = {
            'Overall Churn Rate': self.df['churn'].mean() * 100,
            'High-Risk Customers': len(self.df[
                (self.df['monthly_charges'] > self.df['monthly_charges'].median()) &
                (self.df['daily_viewing_hours'] < self.df['daily_viewing_hours'].median())
            ]) / len(self.df) * 100,
            'Avg Monthly Revenue': self.df['monthly_charges'].mean(),
            'Potential Savings': self.df['churn'].mean() * len(self.df) * self.df['monthly_charges'].mean() * 6
        }
        
        ax1.barh(list(metrics.keys()), list(metrics.values()), 
                color=self.style_config['color_palette'])
        ax1.set_title('Key Business Metrics\nPerformance Overview', fontweight='bold', fontsize=14)
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value annotations
        for i, (key, value) in enumerate(metrics.items()):
            ax1.text(value + max(metrics.values())*0.01, i, 
                   f'${value:,.0f}' if 'Revenue' in key or 'Savings' in key else f'{value:.1f}%', 
                   va='center', fontweight='bold')
        
        # 2. Action Priority Matrix
        self._create_action_priority_matrix(ax2)
        
        # 3. ROI Projection
        self._create_roi_projection(ax3)
        
        # 4. Recommendation Impact
        self._create_recommendation_impact(ax4)
        
        plt.suptitle('Executive Summary: Data-Driven Retention Strategy\nGlobalStream Inc. Customer Churn Analysis', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.savefig(f'{save_path}/executive_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Executive summary chart saved to {save_path}")
    
    def _create_action_priority_matrix(self, ax):
        """Create action priority matrix"""
        # Define initiatives and their impact/effort
        initiatives = {
            'Targeted Discounts': {'impact': 8, 'effort': 3},
            'Usage Alerts': {'impact': 6, 'effort': 2},
            'Loyalty Program': {'impact': 7, 'effort': 7},
            'Personalized Content': {'impact': 5, 'effort': 4},
            'Price Optimization': {'impact': 9, 'effort': 6}
        }
        
        # Create scatter plot
        for initiative, scores in initiatives.items():
            ax.scatter(scores['effort'], scores['impact'], s=200, 
                      alpha=0.7, label=initiative)
            ax.annotate(initiative, (scores['effort'], scores['impact']), 
                       xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        ax.set_xlabel('Implementation Effort (1-10)')
        ax.set_ylabel('Business Impact (1-10)')
        ax.set_title('Action Priority Matrix\nStrategic Initiative Planning', 
                    fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        # Add quadrant labels
        ax.text(2, 8, 'Quick Wins', fontsize=10, fontweight='bold', 
               ha='center', va='center', color='green')
        ax.text(8, 8, 'Major Projects', fontsize=10, fontweight='bold', 
               ha='center', va='center', color='blue')
        ax.text(2, 2, 'Fill-Ins', fontsize=10, fontweight='bold', 
               ha='center', va='center', color='gray')
        ax.text(8, 2, 'Thankless Tasks', fontsize=10, fontweight='bold', 
               ha='center', va='center', color='red')
    
    def _create_roi_projection(self, ax):
        """Create ROI projection chart"""
        months = np.arange(1, 13)
        current_churn = self.df['churn'].mean()
        
        # Projection scenarios
        baseline = [current_churn * 100] * 12
        conservative = [current_churn * 100 * (0.95 ** i) for i in range(12)]
        targeted = [current_churn * 100 * (0.85 ** i) for i in range(12)]
        
        ax.plot(months, baseline, label='Current Trajectory', 
               color='red', linewidth=2, marker='o')
        ax.plot(months, conservative, label='Conservative Strategy', 
               color='orange', linewidth=2, marker='s')
        ax.plot(months, targeted, label='Targeted Interventions', 
               color='green', linewidth=2, marker='^')
        
        ax.set_xlabel('Months')
        ax.set_ylabel('Churn Rate (%)')
        ax.set_title('ROI Projection: Churn Reduction Scenarios\nFinancial Impact Forecast', 
                    fontweight='bold', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(months)
    
    def _create_recommendation_impact(self, ax):
        """Create recommendation impact chart"""
        recommendations = [
            'Price Optimization',
            'Proactive Retention', 
            'Usage Improvement',
            'Loyalty Program',
            'Personalization'
        ]
        
        impact_scores = [8.5, 7.8, 6.2, 5.9, 5.5]
        implementation_time = [6, 3, 4, 8, 5]  # months
        
        bars = ax.barh(recommendations, impact_scores, 
                      color=self.style_config['color_palette'])
        
        ax.set_xlabel('Business Impact Score (1-10)')
        ax.set_title('Recommended Initiatives\nImpact Assessment', 
                    fontweight='bold', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        
        # Add impact scores and implementation time
        for i, (bar, impact, time) in enumerate(zip(bars, impact_scores, implementation_time)):
            ax.text(impact + 0.1, i, f'Score: {impact}\nTime: {time}mo', 
                   va='center', fontsize=8, fontweight='bold')


if __name__ == "__main__":
    # Load engineered features
    df = pd.read_csv('data/features/engineered_features.csv')
    
    # Create visualizations
    visualizer = AdvancedVisualizer(df)
    visualizer.create_comprehensive_dashboard()
    visualizer.create_interactive_dashboard()
    visualizer.create_executive_summary_chart()
    
    print("🎉 All visualizations created successfully!")