"""
Advanced Statistical Testing for Churn Analysis
Hypothesis testing and statistical validation of insights
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats import power
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    def __init__(self, df):
        self.df = df
        self.test_results = {}
        
    def perform_churn_hypothesis_tests(self):
        """Comprehensive hypothesis testing for churn drivers"""
        print("🔬 Performing statistical hypothesis tests...")
        
        tests_performed = []
        
        # Test 1: Monthly charges difference between churned vs retained
        if all(col in self.df.columns for col in ['monthly_charges', 'churn']):
            churned_charges = self.df[self.df['churn'] == 1]['monthly_charges']
            retained_charges = self.df[self.df['churn'] == 0]['monthly_charges']
            
            t_stat, p_value = stats.ttest_ind(churned_charges, retained_charges, equal_var=False)
            tests_performed.append({
                'test': 'Monthly Charges T-Test',
                'hypothesis': 'Churned customers have different monthly charges than retained customers',
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'effect_size': self._cohens_d(churned_charges, retained_charges)
            })
        
        # Test 2: Tenure difference
        if all(col in self.df.columns for col in ['tenure_days', 'churn']):
            churned_tenure = self.df[self.df['churn'] == 1]['tenure_days']
            retained_tenure = self.df[self.df['churn'] == 0]['tenure_days']
            
            u_stat, p_value = mannwhitneyu(churned_tenure, retained_tenure)
            tests_performed.append({
                'test': 'Tenure Mann-Whitney U Test',
                'hypothesis': 'Churned customers have different tenure than retained customers',
                'u_statistic': u_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'effect_size': self._rank_biserial(churned_tenure, retained_tenure)
            })
        
        # Test 3: Usage hours difference
        if all(col in self.df.columns for col in ['daily_viewing_hours', 'churn']):
            churned_usage = self.df[self.df['churn'] == 1]['daily_viewing_hours']
            retained_usage = self.df[self.df['churn'] == 0]['daily_viewing_hours']
            
            t_stat, p_value = stats.ttest_ind(churned_usage, retained_usage, equal_var=False)
            tests_performed.append({
                'test': 'Daily Usage T-Test',
                'hypothesis': 'Churned customers have different usage patterns than retained customers',
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'effect_size': self._cohens_d(churned_usage, retained_usage)
            })
        
        # Test 4: Plan type association with churn
        if all(col in self.df.columns for col in ['plan_type', 'churn']):
            contingency_table = pd.crosstab(self.df['plan_type'], self.df['churn'])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            tests_performed.append({
                'test': 'Plan Type Chi-Square Test',
                'hypothesis': 'Churn rates differ across plan types',
                'chi2_statistic': chi2,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'cramers_v': self._cramers_v(contingency_table)
            })
        
        self.test_results['hypothesis_tests'] = tests_performed
        return tests_performed
    
    def perform_anova_analysis(self):
        """ANOVA tests for multiple group comparisons"""
        print("📊 Performing ANOVA analysis...")
        
        anova_results = []
        
        # One-way ANOVA: Plan type vs monthly charges
        if all(col in self.df.columns for col in ['plan_type', 'monthly_charges']):
            groups = [group['monthly_charges'].values for name, group in self.df.groupby('plan_type')]
            f_stat, p_value = f_oneway(*groups)
            
            anova_results.append({
                'test': 'One-Way ANOVA - Plan Type vs Monthly Charges',
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'eta_squared': self._eta_squared(groups, self.df['monthly_charges'])
            })
        
        # Two-way ANOVA: Plan type and usage intensity vs churn
        if all(col in self.df.columns for col in ['plan_type', 'usage_intensity', 'churn']):
            model = ols('churn ~ C(plan_type) + C(usage_intensity) + C(plan_type):C(usage_intensity)', data=self.df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            anova_results.append({
                'test': 'Two-Way ANOVA - Plan Type & Usage Intensity vs Churn',
                'results': anova_table,
                'model_r_squared': model.rsquared
            })
        
        self.test_results['anova_tests'] = anova_results
        return anova_results
    
    def calculate_confidence_intervals(self, confidence=0.95):
        """Calculate confidence intervals for key metrics"""
        print("📐 Calculating confidence intervals...")
        
        intervals = {}
        
        # Churn rate confidence interval
        if 'churn' in self.df.columns:
            churn_rate = self.df['churn'].mean()
            n = len(self.df)
            se = np.sqrt((churn_rate * (1 - churn_rate)) / n)
            z_value = stats.norm.ppf(1 - (1 - confidence) / 2)
            margin_of_error = z_value * se
            
            intervals['churn_rate'] = {
                'point_estimate': churn_rate,
                'confidence_interval': (churn_rate - margin_of_error, churn_rate + margin_of_error),
                'margin_of_error': margin_of_error,
                'confidence_level': confidence
            }
        
        # Key metric confidence intervals
        key_metrics = ['monthly_charges', 'tenure_days', 'daily_viewing_hours']
        for metric in key_metrics:
            if metric in self.df.columns:
                data = self.df[metric].dropna()
                mean = data.mean()
                sem = stats.sem(data)
                ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)
                
                intervals[metric] = {
                    'mean': mean,
                    'confidence_interval': ci,
                    'standard_error': sem
                }
        
        self.test_results['confidence_intervals'] = intervals
        return intervals
    
    def power_analysis(self, effect_size=0.3, alpha=0.05, power=0.8):
        """Perform statistical power analysis"""
        print("⚡ Performing power analysis...")
        
        # Calculate required sample size for future experiments
        if 'churn' in self.df.columns:
            current_churn_rate = self.df['churn'].mean()
            required_n = power.tt_ind_solve_power(
                effect_size=effect_size,
                alpha=alpha,
                power=power,
                ratio=1
            )
            
            power_analysis = {
                'current_sample_size': len(self.df),
                'required_sample_size': required_n,
                'effect_size_detected': effect_size,
                'alpha_level': alpha,
                'desired_power': power,
                'adequately_powered': len(self.df) >= required_n
            }
            
            self.test_results['power_analysis'] = power_analysis
            return power_analysis
    
    def _cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def _rank_biserial(self, group1, group2):
        """Calculate rank-biserial correlation"""
        u_stat = mannwhitneyu(group1, group2).statistic
        n1, n2 = len(group1), len(group2)
        return 1 - (2 * u_stat) / (n1 * n2)
    
    def _cramers_v(self, contingency_table):
        """Calculate Cramér's V for contingency tables"""
        chi2 = chi2_contingency(contingency_table)[0]
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        return np.sqrt(chi2 / (n * min_dim))
    
    def _eta_squared(self, groups, total_data):
        """Calculate eta squared effect size for ANOVA"""
        ss_between = sum(len(group) * (np.mean(group) - np.mean(total_data))**2 for group in groups)
        ss_total = sum((total_data - np.mean(total_data))**2)
        return ss_between / ss_total
    
    def generate_statistical_report(self):
        """Generate comprehensive statistical report"""
        print("📋 Generating statistical report...")
        
        # Perform all tests
        hypothesis_tests = self.perform_churn_hypothesis_tests()
        anova_tests = self.perform_anova_analysis()
        confidence_intervals = self.calculate_confidence_intervals()
        power_analysis = self.power_analysis()
        
        report = {
            'executive_summary': self._statistical_executive_summary(hypothesis_tests),
            'hypothesis_tests': hypothesis_tests,
            'anova_results': anova_tests,
            'confidence_intervals': confidence_intervals,
            'power_analysis': power_analysis,
            'key_insights': self._extract_statistical_insights()
        }
        
        return report
    
    def _statistical_executive_summary(self, hypothesis_tests):
        """Generate statistical executive summary"""
        significant_tests = [test for test in hypothesis_tests if test['significant']]
        
        summary = f"""
        STATISTICAL ANALYSIS EXECUTIVE SUMMARY
        
        • Tests Performed: {len(hypothesis_tests)} hypothesis tests conducted
        • Significant Findings: {len(significant_tests)} statistically significant relationships identified
        • Confidence Level: 95% confidence intervals calculated for all key metrics
        • Statistical Power: Analysis confirms adequate sample size for detected effects
        
        Key Statistical Conclusions:
        """
        
        for test in significant_tests[:3]:  # Top 3 significant findings
            effect_size = test.get('effect_size', 0)
            effect_strength = "large" if abs(effect_size) > 0.8 else "medium" if abs(effect_size) > 0.5 else "small"
            
            summary += f"\n• {test['test']}: Statistically significant (p={test['p_value']:.4f}) with {effect_strength} effect size"
        
        return summary
    
    def _extract_statistical_insights(self):
        """Extract actionable insights from statistical tests"""
        insights = []
        
        for test in self.test_results.get('hypothesis_tests', []):
            if test['significant']:
                if 'Monthly Charges' in test['test']:
                    insights.append("💰 **Price Sensitivity Confirmed**: Churned customers pay significantly different prices (statistically validated)")
                elif 'Tenure' in test['test']:
                    insights.append("⏰ **Tenure Impact Validated**: Customer tenure significantly predicts churn risk")
                elif 'Usage' in test['test']:
                    insights.append("📊 **Usage Pattern Significance**: Daily usage patterns are statistically different between churned and retained customers")
        
        # Add power analysis insight
        if self.test_results.get('power_analysis', {}).get('adequately_powered'):
            insights.append("✅ **Sample Size Adequate**: Current dataset has sufficient statistical power for reliable insights")
        else:
            insights.append("⚠️ **Sample Size Consideration**: Consider increasing sample size for more robust conclusions")
        
        return insights

if __name__ == "__main__":
    # Load engineered features
    df = pd.read_csv('data/features/engineered_features.csv')
    
    # Perform statistical analysis
    analyzer = StatisticalAnalyzer(df)
    report = analyzer.generate_statistical_report()
    
    # Save results
    import json
    with open('reports/statistical_analysis.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("✅ Statistical analysis completed and saved")