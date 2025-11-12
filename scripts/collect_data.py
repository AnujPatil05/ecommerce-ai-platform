"""
Data Collection Script for E-commerce AI Platform
This script downloads and organizes all required datasets
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
from typing import Dict, List
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCollector:
    """Handles downloading and organizing datasets"""
    
    def __init__(self, base_path: str = "data/raw"):
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            "brazilian_ecommerce": {
                "name": "Brazilian E-commerce Dataset",
                "kaggle_id": "olistbr/brazilian-ecommerce",
                "files": [
                    "olist_customers_dataset.csv",
                    "olist_geolocation_dataset.csv",
                    "olist_order_items_dataset.csv",
                    "olist_order_payments_dataset.csv",
                    "olist_order_reviews_dataset.csv",
                    "olist_orders_dataset.csv",
                    "olist_products_dataset.csv",
                    "olist_sellers_dataset.csv",
                    "product_category_name_translation.csv"
                ]
            },
            "fraud_detection": {
                "name": "Credit Card Fraud Detection",
                "kaggle_id": "mlg-ulb/creditcardfraud",
                "files": ["creditcard.csv"]
            },
            # "amazon_reviews": {
            #     "name": "Amazon Fine Food Reviews",
            #     "kaggle_id": "snap/amazon-fine-food-reviews",
            #     "files": ["Reviews.csv"]
            # },
            # "customer_behavior": {
            #     "name": "E-commerce Customer Behavior",
            #     "kaggle_id": "mkechinov/ecommerce-behavior-data-from-multi-category-store",
            #     "files": ["2019-Oct.csv", "2019-Nov.csv"]
            # }
        }
    
    def check_kaggle_setup(self) -> bool:
        """Check if Kaggle API is properly configured"""
        kaggle_path = Path.home() / ".kaggle" / "kaggle.json"
        
        if not kaggle_path.exists():
            logger.error("Kaggle API credentials not found!")
            logger.info("Please follow these steps:")
            logger.info("1. Go to https://www.kaggle.com/account")
            logger.info("2. Scroll to 'API' section")
            logger.info("3. Click 'Create New API Token'")
            logger.info("4. Move downloaded kaggle.json to ~/.kaggle/")
            logger.info("5. Run: chmod 600 ~/.kaggle/kaggle.json")
            return False
        
        try:
            import kaggle
            logger.info("✓ Kaggle API is configured")
            return True
        except ImportError:
            logger.error("Kaggle package not installed")
            logger.info("Install with: pip install kaggle")
            return False
    
    def download_from_kaggle(self, dataset_id: str, output_path: Path) -> bool:
        """Download dataset from Kaggle"""
        try:
            import kaggle
            
            logger.info(f"Downloading {dataset_id}...")
            output_path.mkdir(parents=True, exist_ok=True)
            
            kaggle.api.dataset_download_files(
                dataset_id,
                path=str(output_path),
                unzip=True,
                quiet=False
            )
            
            logger.info(f"✓ Successfully downloaded {dataset_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {dataset_id}: {str(e)}")
            return False
    
    def download_all_datasets(self):
        """Download all required datasets"""
        if not self.check_kaggle_setup():
            return
        
        logger.info("Starting dataset downloads...")
        logger.info("=" * 60)
        
        for key, dataset in self.datasets.items():
            logger.info(f"\n📦 {dataset['name']}")
            logger.info("-" * 60)
            
            output_path = self.base_path / key
            
            # Check if already downloaded
            if output_path.exists() and any(output_path.iterdir()):
                logger.info(f"⚠ Dataset already exists at {output_path}")
                response = input("Download again? (y/n): ")
                if response.lower() != 'y':
                    logger.info("Skipping...")
                    continue
            
            success = self.download_from_kaggle(
                dataset['kaggle_id'],
                output_path
            )
            
            if success:
                logger.info(f"✓ Saved to: {output_path}")
            else:
                logger.error(f"✗ Failed to download {dataset['name']}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Data collection completed!")
    
    def verify_downloads(self) -> Dict[str, bool]:
        """Verify all datasets are downloaded correctly"""
        logger.info("\n🔍 Verifying downloads...")
        verification = {}
        
        for key, dataset in self.datasets.items():
            dataset_path = self.base_path / key
            
            if not dataset_path.exists():
                verification[key] = False
                logger.warning(f"✗ {dataset['name']}: Not found")
                continue
            
            # Check if expected files exist
            files_exist = []
            for file in dataset['files']:
                file_path = dataset_path / file
                files_exist.append(file_path.exists())
            
            verification[key] = all(files_exist)
            
            if verification[key]:
                logger.info(f"✓ {dataset['name']}: All files present")
            else:
                logger.warning(f"⚠ {dataset['name']}: Some files missing")
        
        return verification
    
    def create_directory_structure(self):
        """Create additional directory structure for processed data"""
        directories = [
            "data/processed/brazilian_ecommerce",
            "data/processed/fraud_detection",
            # "data/processed/amazon_reviews",
            # "data/processed/customer_behavior",
            "data/interim",
            "data/external",
            "models",
            "logs",
            "notebooks/eda",
            "notebooks/modeling",
            "notebooks/evaluation"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            # Create .gitkeep to track empty directories
            gitkeep = Path(directory) / ".gitkeep"
            gitkeep.touch()
        
        logger.info("✓ Directory structure created")
    
    def generate_data_summary(self):
        """Generate a summary of downloaded datasets"""
        try:
            import pandas as pd
        except ImportError:
            logger.error("Pandas not found. Skipping data summary.")
            logger.info("Install with: pip install pandas")
            return

        summary = []
        
        for key, dataset in self.datasets.items():
            dataset_path = self.base_path / key
            
            if not dataset_path.exists():
                continue
            
            for file in dataset['files']:
                file_path = dataset_path / file
                
                if file_path.exists() and file_path.suffix == '.csv':
                    try:
                        df = pd.read_csv(file_path, nrows=5)
                        
                        # Get file size
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        
                        summary.append({
                            'Dataset': dataset['name'],
                            'File': file,
                            'Rows': 'Loading...',
                            'Columns': len(df.columns),
                            'Size (MB)': f"{size_mb:.2f}",
                            'Path': str(file_path.relative_to(Path.cwd()))
                        })
                    except Exception as e:
                        logger.warning(f"Could not read {file}: {str(e)}")
        
        if summary:
            summary_df = pd.DataFrame(summary)
            summary_path = self.base_path / "data_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            
            logger.info("\n" + "=" * 80)
            logger.info("📊 DATA SUMMARY")
            logger.info("=" * 80)
            print(summary_df.to_string(index=False))
            logger.info(f"\n✓ Summary saved to: {summary_path}")


def main():
    """Main execution function"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        E-COMMERCE AI PLATFORM - DATA COLLECTOR               ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    collector = DataCollector()
    
    # Create directory structure
    logger.info("Setting up directory structure...")
    collector.create_directory_structure()
    
    # Download datasets
    print("\n" + "=" * 60)
    print("DATASET DOWNLOAD")
    print("=" * 60)
    print("\nThis will download data from Kaggle")
    print("Make sure you have:")
    print("  1. Kaggle account")
    print("  2. API credentials configured (~/.kaggle/kaggle.json)")
    print("  3. Accepted terms for each dataset on Kaggle")
    print("\n" + "=" * 60 + "\n")
    
    response = input("Proceed with download? (y/n): ")
    
    if response.lower() == 'y':
        collector.download_all_datasets()
        
        # Verify downloads
        verification = collector.verify_downloads()
        
        # Generate summary
        collector.generate_data_summary()
        
        # Show next steps
        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("\n1. Review data summary in data/raw/data_summary.csv")
        print("2. Run EDA notebook: notebooks/01_data_collection.ipynb")
        print("3. Start data preprocessing: python scripts/preprocess_data.py")
        print("\n" + "=" * 80)
    else:
        print("\nDownload cancelled.")


if __name__ == "__main__":
    main()