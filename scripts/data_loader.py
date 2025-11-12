"""
Data Loading Utilities for E-commerce AI Platform
Handles loading and initial processing of all datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DataLoader:
    """Centralized data loading for all datasets"""
    
    def __init__(self, raw_data_path: str = "data/raw"):
        self.raw_path = Path(raw_data_path)
        self.processed_path = Path("data/processed")
        
        # Cache loaded dataframes
        self._cache = {}
    
    # ==================== BRAZILIAN E-COMMERCE ====================
    
    def load_brazilian_ecommerce(self, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load all Brazilian e-commerce datasets
        
        Returns:
            Dictionary of dataframes with keys:
            - orders, order_items, customers, products, sellers, 
              reviews, payments, geolocation, category_translation
        """
        cache_key = "brazilian_ecommerce"
        
        if use_cache and cache_key in self._cache:
            logger.info("Loading from cache...")
            return self._cache[cache_key]
        
        base_path = self.raw_path / "brazilian_ecommerce"
        
        logger.info("Loading Brazilian E-commerce datasets...")
        
        datasets = {
            'orders': self._load_csv(base_path / "olist_orders_dataset.csv"),
            'order_items': self._load_csv(base_path / "olist_order_items_dataset.csv"),
            'customers': self._load_csv(base_path / "olist_customers_dataset.csv"),
            'products': self._load_csv(base_path / "olist_products_dataset.csv"),
            'sellers': self._load_csv(base_path / "olist_sellers_dataset.csv"),
            'reviews': self._load_csv(base_path / "olist_order_reviews_dataset.csv"),
            'payments': self._load_csv(base_path / "olist_order_payments_dataset.csv"),
            'geolocation': self._load_csv(base_path / "olist_geolocation_dataset.csv"),
            'category_translation': self._load_csv(
                base_path / "product_category_name_translation.csv"
            )
        }
        
        # Convert datetime columns
        date_columns = {
            'orders': [
                'order_purchase_timestamp',
                'order_approved_at',
                'order_delivered_carrier_date',
                'order_delivered_customer_date',
                'order_estimated_delivery_date'
            ],
            'reviews': [
                'review_creation_date',
                'review_answer_timestamp'
            ],
            'order_items': [
                'shipping_limit_date'
            ]
        }
        
        for df_name, cols in date_columns.items():
            if df_name in datasets:
                for col in cols:
                    if col in datasets[df_name].columns:
                        datasets[df_name][col] = pd.to_datetime(
                            datasets[df_name][col], 
                            errors='coerce'
                        )
        
        self._cache[cache_key] = datasets
        
        logger.info(f"✓ Loaded {len(datasets)} Brazilian E-commerce tables")
        self._print_summary(datasets)
        
        return datasets
    
    # ==================== FRAUD DETECTION ====================
    
    def load_fraud_data(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load credit card fraud detection dataset
        
        Returns:
            DataFrame with transaction data
        """
        cache_key = "fraud_detection"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        logger.info("Loading Fraud Detection dataset...")
        
        file_path = self.raw_path / "fraud_detection" / "creditcard.csv"
        df = self._load_csv(file_path)
        
        # Basic preprocessing
        if 'Time' in df.columns:
            # Convert time to hours
            df['Hour'] = (df['Time'] / 3600) % 24
            df['Day'] = (df['Time'] / 86400).astype(int)
        
        self._cache[cache_key] = df
        
        logger.info(f"✓ Loaded Fraud Detection data: {df.shape}")
        if 'Class' in df.columns:
            logger.info(f"  - Normal transactions: {(df['Class'] == 0).sum():,}")
            logger.info(f"  - Fraudulent transactions: {(df['Class'] == 1).sum():,}")
            logger.info(f"  - Fraud rate: {df['Class'].mean() * 100:.4f}%")
        
        return df
    
    # ==================== UTILITY METHODS ====================
    
    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """Load CSV with error handling"""
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise
    
    def _print_summary(self, datasets: Dict[str, pd.DataFrame]):
        """Print summary of loaded datasets"""
        for name, df in datasets.items():
            logger.info(f"  - {name}: {df.shape}")
    
    def get_data_info(self) -> Dict:
        """Get information about all available datasets"""
        info = {
            'brazilian_ecommerce': {
                'tables': 9,
                'description': 'Complete e-commerce transaction data',
                'size': '~100K orders'
            },
            'fraud_detection': {
                'tables': 1,
                'description': 'Credit card fraud transactions',
                'size': '~284K transactions'
            }
        }
        return info
    
    def clear_cache(self):
        """Clear all cached data"""
        self._cache.clear()
        logger.info("Cache cleared")


# ==================== CONVENIENCE FUNCTIONS ====================

def quick_load_for_eda() -> Dict[str, pd.DataFrame]:
    """
    Quick load of sampled data for initial EDA
    Returns smaller datasets suitable for exploration
    """
    loader = DataLoader()
    
    data = {
        'ecommerce': loader.load_brazilian_ecommerce(),
        'fraud': loader.load_fraud_data(),
    }
    
    return data


def load_for_modeling(dataset_name: str) -> pd.DataFrame:
    """
    Load full dataset for modeling
    
    Args:
        dataset_name: One of 'ecommerce', 'fraud'
    """
    loader = DataLoader()
    
    if dataset_name == 'fraud':
        return loader.load_fraud_data()
    elif dataset_name == 'ecommerce':
        return loader.load_brazilian_ecommerce()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == "__main__":
    # Test data loading
    logging.basicConfig(level=logging.INFO)
    
    loader = DataLoader()
    
    print("\n" + "="*60)
    print("TESTING DATA LOADER")
    print("="*60 + "\n")
    
    # Test each dataset
    print("\n1. Testing Brazilian E-commerce...")
    ecom = loader.load_brazilian_ecommerce()
    
    print("\n2. Testing Fraud Detection...")
    fraud = loader.load_fraud_data()
    
    print("\n" + "="*60)
    print("✓ All datasets loaded successfully!")
    print("="*60)