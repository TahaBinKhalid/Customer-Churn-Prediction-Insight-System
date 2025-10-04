"""
Advanced Feature Engineering for Churn Prediction
Creates meaningful predictors from raw data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self, df):
        self.df = df
        self.feature_importance = {}
        
    def create_temporal_features(self):
        """Create time-based features from date columns"""
        print("🕐 Creating temporal features...")
        
        if 'signup_date' in self.df.columns:
            # Customer tenure in days
            self.df['tenure_days'] = (pd.to_datetime('today') - self.df['signup_date']).dt.days
            
            # Days since last login
            if 'last_login' in self.df.columns:
                self.df['days_since_login'] = (pd.to_datetime('today') - self.df['last_login']).dt.days
        
        # Monthly usage patterns
        if 'viewing_hours' in self.df.columns and 'tenure_days' in self.df.columns:
            self.df['daily_viewing_hours'] = self.df['viewing_hours'] / np.maximum(self.df['tenure_days'], 1)
        
        return self.df
    
    def create_behavioral_features(self):
        """Create behavioral and engagement metrics"""
        print("📈 Creating behavioral features...")
        
        # Usage intensity
        if 'viewing_hours' in self.df.columns:
            self.df['usage_intensity'] = pd.cut(self.df['viewing_hours'], 
                                               bins=[0, 10, 30, 100, np.inf],
                                               labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Payment efficiency
        if 'monthly_charges' in self.df.columns and 'total_charges' in self.df.columns:
            self.df['payment_value_ratio'] = self.df['monthly_charges'] / (self.df['total_charges'] + 1)
        
        # Device utilization
        if 'devices_connected' in self.df.columns:
            self.df['device_utilization'] = self.df['devices_connected'] / 5.0  # Assuming 5 is max
        
        return self.df
    
    def create_interaction_features(self):
        """Create interaction terms between important variables"""
        print("🔄 Creating interaction features...")
        
        # Interaction between tenure and usage
        if 'tenure_days' in self.df.columns and 'viewing_hours' in self.df.columns:
            self.df['tenure_usage_interaction'] = self.df['tenure_days'] * self.df['viewing_hours']
        
        # Price sensitivity indicator
        if 'monthly_charges' in self.df.columns and 'daily_viewing_hours' in self.df.columns:
            self.df['price_per_viewing_hour'] = self.df['monthly_charges'] / (self.df['daily_viewing_hours'] + 0.1)
        
        return self.df
    
    def encode_categorical_features(self):
        """Encode categorical variables appropriately"""
        print("🔤 Encoding categorical features...")
        
        # One-hot encoding for low cardinality features
        low_cardinality_cols = ['usage_intensity', 'plan_type', 'payment_method']
        existing_low_card = [col for col in low_cardinality_cols if col in self.df.columns]
        
        if existing_low_card:
            self.df = pd.get_dummies(self.df, columns=existing_low_card, prefix=existing_low_card)
        
        # Label encoding for high cardinality features
        high_cardinality_cols = ['country', 'device_type']
        existing_high_card = [col for col in high_cardinality_cols if col in self.df.columns]
        
        for col in existing_high_card:
            le = LabelEncoder()
            self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
        
        return self.df
    
    def select_best_features(self, target_column='churn', k=15):
        """Select most predictive features using statistical tests"""
        print(f"🎯 Selecting top {k} features...")
        
        # Prepare data for feature selection
        X = self.df.select_dtypes(include=[np.number]).drop(columns=[target_column], errors='ignore')
        y = self.df[target_column]
        
        # Remove columns with zero variance
        X = X.loc[:, X.std() > 0]
        
        # Perform feature selection
        selector = SelectKBest(score_func=f_classif, k=min(k, len(X.columns)))
        selector.fit(X, y)
        
        # Get feature scores
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_,
            'p_value': selector.pvalues_
        }).sort_values('score', ascending=False)
        
        self.feature_importance = feature_scores.set_index('feature')['score'].to_dict()
        
        # Select top features
        selected_features = feature_scores.head(k)['feature'].tolist()
        self.df = self.df[selected_features + [target_column]]
        
        print("✅ Feature selection completed")
        return self.df, feature_scores
    
    def get_feature_importance_report(self):
        """Generate feature importance report"""
        if self.feature_importance:
            importance_df = pd.DataFrame.from_dict(self.feature_importance, 
                                                 orient='index', 
                                                 columns=['importance_score'])
            return importance_df.sort_values('importance_score', ascending=False)
        else:
            print("⚠️  No feature importance data available. Run select_best_features first.")
            return None

# Example usage
if __name__ == "__main__":
    # Load cleaned data
    df = pd.read_pickle('data/processed/cleaned_data.pkl')
    
    # Engineer features
    engineer = FeatureEngineer(df)
    engineer.create_temporal_features()
    engineer.create_behavioral_features()
    engineer.create_interaction_features()
    engineer.encode_categorical_features()
    
    # Select best features
    final_df, feature_scores = engineer.select_best_features(target_column='churn', k=15)
    
    # Save results
    final_df.to_csv('data/features/engineered_features.csv', index=False)
    feature_scores.to_csv('data/features/feature_importance.csv', index=False)