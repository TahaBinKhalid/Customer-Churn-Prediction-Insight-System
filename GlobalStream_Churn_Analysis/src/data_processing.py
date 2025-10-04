"""
Data Processing Module for Customer Churn Analysis
Handles data cleaning, validation, and preparation
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        
    def load_data(self):
        """Load and validate raw customer data"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def clean_data(self):
        """Comprehensive data cleaning pipeline"""
        print("🔄 Starting data cleaning process...")
        
        # Remove duplicates
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        print(f"   Removed {initial_count - len(self.df)} duplicate records")
        
        # Handle missing values
        self._handle_missing_values()
        
        # Fix data types
        self._fix_data_types()
        
        # Validate data ranges
        self._validate_data_ranges()
        
        print("✅ Data cleaning completed")
        return self.df
    
    def _handle_missing_values(self):
        """Strategic handling of missing data"""
        missing_report = self.df.isnull().sum()
        missing_percent = (missing_report / len(self.df)) * 100
        
        print("\n📊 Missing Value Report:")
        for col, percent in missing_percent.items():
            if percent > 0:
                print(f"   {col}: {percent:.1f}% missing")
        
        # Strategy: Remove rows with >50% missing, impute others
        cols_to_drop = missing_percent[missing_percent > 50].index
        self.df = self.df.drop(columns=cols_to_drop)
        
        # Impute numerical columns with median
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numerical_cols] = self.df[numerical_cols].fillna(self.df[numerical_cols].median())
        
        # Impute categorical columns with mode
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown')
    
    def _fix_data_types(self):
        """Ensure correct data types"""
        # Convert date columns
        date_columns = ['signup_date', 'last_login', 'subscription_date']
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        # Ensure numerical columns are properly typed
        numerical_columns = ['monthly_charges', 'total_charges', 'viewing_hours', 'devices_connected']
        for col in numerical_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
    
    def _validate_data_ranges(self):
        """Validate data integrity and ranges"""
        # Check for negative values where inappropriate
        positive_columns = ['monthly_charges', 'total_charges', 'viewing_hours']
        for col in positive_columns:
            if col in self.df.columns:
                negative_count = (self.df[col] < 0).sum()
                if negative_count > 0:
                    print(f"   ⚠️  Found {negative_count} negative values in {col}, converting to absolute")
                    self.df[col] = self.df[col].abs()
        
        # Validate churn flag (should be 0 or 1)
        if 'churn' in self.df.columns:
            invalid_churn = ~self.df['churn'].isin([0, 1])
            if invalid_churn.sum() > 0:
                print(f"   ⚠️  Found {invalid_churn.sum()} invalid churn values, removing these rows")
                self.df = self.df[~invalid_churn]
    
    def save_cleaned_data(self, output_path):
        """Save processed data for analysis"""
        self.df.to_pickle(output_path)
        print(f"💾 Cleaned data saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    processor = DataProcessor('data/raw/customer_data.csv')
    if processor.load_data():
        cleaned_df = processor.clean_data()
        processor.save_cleaned_data('data/processed/cleaned_data.pkl')