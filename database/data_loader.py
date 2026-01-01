"""
Data Loader - Loads simulated data into the database
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_sources.data_simulation import DataSimulator
from database.db_manager import DatabaseManager
from utils.logger import logger

def load_data_to_database():
    """Generate and load data into database"""
    logger.info("Starting data generation...")
    
    # Generate data
    simulator = DataSimulator(start_date="2015-01-01", end_date="2025-01-01")
    data = simulator.generate_all_data()
    
    logger.info("Data generation complete. Loading into database...")
    
    # Initialize database
    db = DatabaseManager()
    
    # Load each domain
    for domain, df in data.items():
        table_name = f"{domain}_data"
        logger.info(f"Loading {len(df)} records into {table_name}...")
        db.insert_dataframe(df, table_name, if_exists='replace')
        logger.info(f"✓ {table_name} loaded successfully")
    
    logger.info("All data loaded successfully!")
    
    # Verify
    for domain in ['climate', 'health', 'food', 'economic']:
        table_name = f"{domain}_data"
        count = db.query_to_dataframe(f"SELECT COUNT(*) as count FROM {table_name}")
        logger.info(f"{table_name}: {count['count'].values[0]} records")
    
    db.close()

if __name__ == "__main__":
    load_data_to_database()
