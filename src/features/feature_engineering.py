"""
Feature Engineering Module for E-commerce AI Platform
Creates features for Recommendation System, Fraud Detection, and Sentiment Analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Main feature engineering class"""

    def __init__(self):
        self.scalers = {}
        self.encoders = {}

    # ==================== RECOMMENDATION FEATURES ====================

    def create_user_features(self, orders: pd.DataFrame, order_items: pd.DataFrame,
                             reviews: pd.DataFrame) -> pd.DataFrame:
        """
        Create user behavior features for recommendation system
        """
        logger.info("Creating user features...")

        # Merge necessary data
        orders_with_items = orders.merge(order_items, on='order_id', how='left')
        orders_with_reviews = orders.merge(reviews[['order_id', 'review_score']],
                                           on='order_id', how='left')

        # Calculate reference date (latest date in dataset)
        reference_date = orders['order_purchase_timestamp'].max()

        user_features = []
        
        # Use vectorized operations instead of loops for speed
        logger.info("Calculating RFM...")
        
        # Recency
        recency_df = orders.groupby('customer_id')['order_purchase_timestamp'].max().reset_index()
        recency_df['recency'] = (reference_date - recency_df['order_purchase_timestamp']).dt.days
        
        # Frequency
        frequency_df = orders.groupby('customer_id')['order_id'].nunique().reset_index()
        frequency_df.columns = ['customer_id', 'frequency']
        
        # Monetary
        monetary_df = orders_with_items.groupby('customer_id')['price'].sum().reset_index()
        monetary_df.columns = ['customer_id', 'monetary']
        
        # Merge RFM
        user_features_df = recency_df[['customer_id', 'recency']].merge(frequency_df, on='customer_id', how='left')
        user_features_df = user_features_df.merge(monetary_df, on='customer_id', how='left')
        
        logger.info("Calculating purchase patterns...")
        # Purchase Patterns
        user_features_df['avg_order_value'] = user_features_df['monetary'] / user_features_df['frequency']
        
        items_count_df = orders_with_items.groupby('customer_id')['product_id'].count().reset_index()
        items_count_df.columns = ['customer_id', 'total_items']
        user_features_df = user_features_df.merge(items_count_df, on='customer_id', how='left')
        
        user_features_df['avg_items_per_order'] = user_features_df['total_items'] / user_features_df['frequency']
        
        # Preferred Hour/Day
        preferred_hour_df = orders.groupby('customer_id')['order_purchase_timestamp'].apply(lambda x: x.dt.hour.mode()[0]).reset_index()
        preferred_hour_df.columns = ['customer_id', 'preferred_hour']
        user_features_df = user_features_df.merge(preferred_hour_df, on='customer_id', how='left')
        
        preferred_day_df = orders.groupby('customer_id')['order_purchase_timestamp'].apply(lambda x: x.dt.dayofweek.mode()[0]).reset_index()
        preferred_day_df.columns = ['customer_id', 'preferred_day']
        user_features_df = user_features_df.merge(preferred_day_df, on='customer_id', how='left')
        
        logger.info("Calculating review behavior...")
        # Review Behavior
        review_stats_df = orders_with_reviews.groupby('customer_id')['review_score'].agg(['mean', 'count']).reset_index()
        review_stats_df.columns = ['customer_id', 'avg_review_score', 'review_count']
        user_features_df = user_features_df.merge(review_stats_df, on='customer_id', how='left')
        user_features_df['review_rate'] = user_features_df['review_count'] / user_features_df['frequency']
        
        logger.info("Calculating product diversity...")
        # Diversity
        diversity_df = orders_with_items.groupby('customer_id')['product_id'].nunique().reset_index()
        diversity_df.columns = ['customer_id', 'unique_products']
        user_features_df = user_features_df.merge(diversity_df, on='customer_id', how='left')
        user_features_df['product_diversity'] = user_features_df['unique_products'] / user_features_df['total_items']

        # Time between purchases
        logger.info("Calculating time between purchases...")
        orders_sorted = orders[['customer_id', 'order_purchase_timestamp']].sort_values(by=['customer_id', 'order_purchase_timestamp'])
        orders_sorted['time_diff'] = orders_sorted.groupby('customer_id')['order_purchase_timestamp'].diff().dt.days
        avg_time_diff_df = orders_sorted.groupby('customer_id')['time_diff'].mean().reset_index()
        avg_time_diff_df.columns = ['customer_id', 'avg_days_between_purchases']
        user_features_df = user_features_df.merge(avg_time_diff_df, on='customer_id', how='left')
        
        # Fill NaNs created during merges/calculations
        user_features_df.fillna(0, inplace=True) 

        logger.info("Calculating RFM scores...")
        # Create RFM score (1-5 scale)
        user_features_df['recency_score'] = pd.qcut(user_features_df['recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        user_features_df['frequency_score'] = pd.qcut(user_features_df['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        user_features_df['monetary_score'] = pd.qcut(user_features_df['monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5], duplicates='drop')

        user_features_df['rfm_score'] = (
            user_features_df['recency_score'].astype(float) +
            user_features_df['frequency_score'].astype(float) +
            user_features_df['monetary_score'].astype(float)
        ) / 3

        # Customer segmentation based on RFM
        user_features_df['customer_segment'] = pd.cut(
            user_features_df['rfm_score'],
            bins=[0, 2, 3, 4, 5],
            labels=['At Risk', 'Potential', 'Loyal', 'Champion'],
            right=True
        )
        
        # Handle potential NaNs in segmentation
        user_features_df['customer_segment'] = user_features_df['customer_segment'].astype(str).replace('nan', 'New')


        logger.info(f"✓ Created {len(user_features_df)} user feature sets")
        return user_features_df

    def create_product_features(self, products: pd.DataFrame, order_items: pd.DataFrame,
                                reviews: pd.DataFrame, orders: pd.DataFrame,
                                category_translation: pd.DataFrame) -> pd.DataFrame:
        """
        Create product features for recommendation system
        """
        logger.info("Creating product features...")

        # Translate categories
        products_trans = products.merge(category_translation, on='product_category_name', how='left')
        products_trans['product_category_name_english'] = products_trans['product_category_name_english'].fillna('unknown')

        # Popularity Features
        popularity_df = order_items.groupby('product_id').agg(
            total_sold=('order_id', 'count'),
            total_revenue=('price', 'sum'),
            avg_price=('price', 'mean'),
            min_price=('price', 'min'),
            max_price=('price', 'max'),
            price_std=('price', 'std')
        ).reset_index()

        # Review Features
        items_with_reviews = order_items.merge(
            reviews[['order_id', 'review_score']],
            on='order_id',
            how='left'
        )
        review_stats_df = items_with_reviews.groupby('product_id')['review_score'].agg(
            avg_rating='mean',
            review_count='count'
        ).reset_index()

        # Temporal Features
        product_orders = order_items.merge(orders[['order_id', 'order_purchase_timestamp']], on='order_id', how='left')
        temporal_df = product_orders.groupby('product_id')['order_purchase_timestamp'].agg(
            first_sale='min',
            last_sale='max'
        ).reset_index()
        temporal_df['days_on_market'] = (temporal_df['last_sale'] - temporal_df['first_sale']).dt.days
        temporal_df['days_on_market'] = temporal_df['days_on_market'].replace(0, 1) # Avoid division by zero

        # Merge all features
        product_features_df = products_trans.merge(popularity_df, on='product_id', how='left')
        product_features_df = product_features_df.merge(review_stats_df, on='product_id', how='left')
        product_features_df = product_features_df.merge(temporal_df, on='product_id', how='left')
        
        # Calculate derived features
        product_features_df['sales_velocity'] = product_features_df['total_sold'] / product_features_df['days_on_market']
        
        # Physical features
        product_features_df['volume'] = (
            product_features_df['product_length_cm'] *
            product_features_df['product_height_cm'] *
            product_features_df['product_width_cm']
        )

        # Popularity score
        product_features_df['popularity_score'] = (
            product_features_df['total_sold'].rank(pct=True) * 0.5 +
            product_features_df['avg_rating'].fillna(3) / 5 * 0.3 +
            product_features_df['review_count'].rank(pct=True) * 0.2
        )

        # Price category
        product_features_df['price_category'] = pd.qcut(
            product_features_df['avg_price'].fillna(product_features_df['avg_price'].median()),
            q=4,
            labels=['Budget', 'Mid-Range', 'Premium', 'Luxury'],
            duplicates='drop'
        )

        # Fill NaNs
        numeric_cols = product_features_df.select_dtypes(include=np.number).columns
        product_features_df[numeric_cols] = product_features_df[numeric_cols].fillna(0)

        # 2. Fill categorical columns with 'Unknown'
        if 'price_category' in product_features_df.columns:
            # Add 'Unknown' as a valid category
            if isinstance(product_features_df['price_category'].dtype, pd.CategoricalDtype):
                product_features_df['price_category'] = product_features_df['price_category'].cat.add_categories(['Unknown'])

            # Now fill the NaNs
            product_features_df['price_category'] = product_features_df['price_category'].fillna('Unknown')

        # 3. Fill object (string) columns with 'unknown'
        object_cols = product_features_df.select_dtypes(include='object').columns
        product_features_df[object_cols] = product_features_df[object_cols].fillna('unknown')
                
        
        product_features_df.rename(columns={'product_category_name_english': 'category'}, inplace=True)
        
        logger.info(f"✓ Created {len(product_features_df)} product feature sets")
        return product_features_df

    # ==================== FRAUD DETECTION FEATURES ====================

    def create_fraud_features(self, fraud_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for fraud detection model
        """
        logger.info("Creating fraud detection features...")

        df = fraud_data.copy()

        # === Temporal Features ===
        df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

        df['day_sin'] = np.sin(2 * np.pi * df['Day'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['Day'] / 7)

        df['time_of_day'] = pd.cut(
            df['Hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            include_lowest=True
        )
        df['is_weekend'] = (df['Day'] % 7 >= 5).astype(int)

        # === Amount Features ===
        df['amount_log'] = np.log1p(df['Amount'])

        amount_by_hour = df.groupby('Hour')['Amount'].transform('mean')
        df['amount_vs_hour_avg'] = df['Amount'] / (amount_by_hour + 1e-5) # Avoid zero division

        amount_by_day = df.groupby('Day')['Amount'].transform('mean')
        df['amount_vs_day_avg'] = df['Amount'] / (amount_by_day + 1e-5)

        # === Rolling Statistics ===
        df = df.sort_values('Time').reset_index(drop=True)
        df['amount_rolling_mean_10'] = df['Amount'].rolling(window=10, min_periods=1).mean()
        df['amount_rolling_std_10'] = df['Amount'].rolling(window=10, min_periods=1).std().fillna(0)

        df['amount_deviation'] = np.abs(df['Amount'] - df['amount_rolling_mean_10'])
        df['amount_deviation_normalized'] = df['amount_deviation'] / (df['amount_rolling_std_10'] + 1e-5)

        # === V Feature Interactions ===
        v_features = [col for col in df.columns if col.startswith('V')]

        if len(v_features) >= 4:
            df['V1_V2_interaction'] = df['V1'] * df['V2']
            df['V3_V4_interaction'] = df['V3'] * df['V4']

        df['v_sum_abs'] = df[v_features].abs().sum(axis=1)
        df['v_mean'] = df[v_features].mean(axis=1)
        df['v_std'] = df[v_features].std(axis=1)
        

        # 1. Fill numeric columns with 0
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        # 2. Fill categorical column 'time_of_day' with 'Unknown'
        if 'time_of_day' in df.columns:
            # Add 'Unknown' as a valid category
            if isinstance(df['time_of_day'].dtype, pd.CategoricalDtype):
                df['time_of_day'] = df['time_of_day'].cat.add_categories(['Unknown'])

            # Now fill the NaNs
            df['time_of_day'] = df['time_of_day'].fillna('Unknown')


        object_cols = df.select_dtypes(include='object').columns
        df[object_cols] = df[object_cols].fillna('unknown')


        logger.info(f"✓ Created fraud features: {df.shape[1]} total features")
        return df

    # ==================== TEXT FEATURES FOR SENTIMENT ====================

    def create_text_features(self, reviews: pd.DataFrame) -> pd.DataFrame:
        """
        Create text-based features for sentiment analysis
        """
        logger.info("Creating text features...")

        df = reviews.copy()
        df['review_comment_message'] = df['review_comment_message'].fillna('')

        df['review_length'] = df['review_comment_message'].str.len()
        df['word_count'] = df['review_comment_message'].str.split().str.len()
        df['avg_word_length'] = df['review_length'] / (df['word_count'] + 1e-5)

        df['exclamation_count'] = df['review_comment_message'].str.count('!')
        df['question_count'] = df['review_comment_message'].str.count(r'\?')
        df['comma_count'] = df['review_comment_message'].str.count(',')
        df['period_count'] = df['review_comment_message'].str.count(r'\.')

        df['total_punctuation'] = (
            df['exclamation_count'] +
            df['question_count'] +
            df['comma_count'] +
            df['period_count']
        )

        df['uppercase_ratio'] = df['review_comment_message'].apply(
            lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1e-5)
        )
        df['has_review_text'] = (df['review_length'] > 0).astype(int)

        if 'review_comment_title' in df.columns:
            df['title_length'] = df['review_comment_title'].fillna('').str.len()
            df['has_title'] = (df['title_length'] > 0).astype(int)

        if 'review_creation_date' in df.columns:
            df['review_hour'] = df['review_creation_date'].dt.hour
            df['review_day_of_week'] = df['review_creation_date'].dt.dayofweek
            
        df.fillna(0, inplace=True)

        logger.info(f"✓ Created text features: {df.shape[1]} total features")
        return df

    # ==================== HELPER FUNCTIONS ====================

    def scale_features(self, df: pd.DataFrame, features: List[str],
                         method: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features
        """
        df_scaled = df.copy()

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()

        df_scaled[features] = scaler.fit_transform(df_scaled[features])
        self.scalers[f"{method}_scaler_{'_'.join(features)}"] = scaler

        logger.info(f"✓ Scaled {len(features)} features using {method} scaling")
        return df_scaled

    def encode_categorical(self, df: pd.DataFrame, columns: List[str],
                             method: str = 'label') -> pd.DataFrame:
        """
        Encode categorical features
        """
        df_encoded = df.copy()

        if method == 'label':
            for col in columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.encoders[f"{col}_encoder"] = le

        elif method == 'onehot':
            df_encoded = pd.get_dummies(df_encoded, columns=columns, prefix=columns, dummy_na=True)

        logger.info(f"✓ Encoded {len(columns)} categorical features using {method} encoding")
        return df_encoded

    def create_interaction_features(self, df: pd.DataFrame,
                                      feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction features between pairs of features
        """
        df_interactions = df.copy()

        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                df_interactions[f'{feat1}_x_{feat2}'] = df_interactions[feat1] * df_interactions[feat2]
                df_interactions[f'{feat1}_div_{feat2}'] = df_interactions[feat1] / (df_interactions[feat2] + 1e-5)

        logger.info(f"✓ Created {len(feature_pairs) * 2} interaction features")
        return df_interactions


# ==================== CONVENIENCE FUNCTIONS ====================

def prepare_recommendation_features(ecom_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare all features needed for recommendation system
    """
    engineer = FeatureEngineer()

    user_features = engineer.create_user_features(
        ecom_data['orders'],
        ecom_data['order_items'],
        ecom_data['reviews']
    )

    product_features = engineer.create_product_features(
        ecom_data['products'],
        ecom_data['order_items'],
        ecom_data['reviews'],
        ecom_data['orders'],
        ecom_data['category_translation']
    )

    return user_features, product_features


def prepare_fraud_features(fraud_data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare all features needed for fraud detection
    """
    engineer = FeatureEngineer()
    fraud_features = engineer.create_fraud_features(fraud_data)
    return fraud_features


def prepare_sentiment_features(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare all features needed for sentiment analysis
    """
    engineer = FeatureEngineer()
    text_features = engineer.create_text_features(reviews)
    return text_features


if __name__ == "__main__":
    # Test feature engineering
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("="*70)
    logger.info("TESTING FEATURE ENGINEERING")
    logger.info("="*70)

    try:
        # We must import the data loader relative to the project root
        # This assumes you run this script from the project root folder
        # e.g., python src/features/feature_engineering.py
        from scripts.data_loader import DataLoader
        
        loader = DataLoader(raw_data_path='data/raw') # Assuming run from root
        ecom_data = loader.load_brazilian_ecommerce()
        fraud_data = loader.load_fraud_data()

        logger.info("\n--- Testing Recommendation Features ---")
        user_feat, product_feat = prepare_recommendation_features(ecom_data)
        logger.info(f"User features shape: {user_feat.shape}")
        logger.info(f"Product features shape: {product_feat.shape}")

        logger.info("\n--- Testing Fraud Features ---")
        fraud_feat = prepare_fraud_features(fraud_data)
        logger.info(f"Fraud features shape: {fraud_feat.shape}")

        logger.info("\n--- Testing Sentiment Features ---")
        text_feat = prepare_sentiment_features(ecom_data['reviews'])
        logger.info(f"Sentiment features shape: {text_feat.shape}")
        
        logger.info("\n" + "="*70)
        logger.info("✅ FEATURE ENGINEERING TESTS PASSED")
        logger.info("="*70)

    except ImportError:
        logger.error("Could not import DataLoader. Run this test from the project root directory.")
    except Exception as e:
        logger.error(f"An error occurred during testing: {e}")