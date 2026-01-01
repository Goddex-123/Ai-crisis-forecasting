"""
Database Manager for Crisis Forecasting System
Handles all database operations using SQLAlchemy
"""
import sqlite3
from sqlalchemy import create_engine, Table, MetaData
from sqlalchemy.orm import sessionmaker, Session
from pathlib import Path
import pandas as pd
from typing import Optional, List, Dict
import yaml

class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize database manager"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        db_config = self.config['database']
        self.db_path = db_config['path']
        
        # Create engine
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=db_config.get('echo', False))
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.metadata = MetaData()
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        schema_path = Path(__file__).parent / "schema.sql"
        
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            # Execute schema
            with self.engine.connect() as conn:
                for statement in schema_sql.split(';'):
                    if statement.strip():
                        conn.execute(statement)
                conn.commit()
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()
    
    def insert_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append'):
        """Insert pandas DataFrame into table"""
        df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
    
    def query_to_dataframe(self, query: str) -> pd.DataFrame:
        """Execute query and return results as DataFrame"""
        return pd.read_sql_query(query, self.engine)
    
    def get_table_data(self, table_name: str, 
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       region: Optional[str] = None) -> pd.DataFrame:
        """Get data from table with optional filters"""
        query = f"SELECT * FROM {table_name} WHERE 1=1"
        
        if start_date:
            query += f" AND date >= '{start_date}'"
        if end_date:
            query += f" AND date <= '{end_date}'"
        if region:
            query += f" AND region = '{region}'"
        
        query += " ORDER BY date"
        
        return self.query_to_dataframe(query)
    
    def get_latest_data(self, table_name: str, limit: int = 100) -> pd.DataFrame:
        """Get latest records from table"""
        query = f"SELECT * FROM {table_name} ORDER BY date DESC LIMIT {limit}"
        return self.query_to_dataframe(query)
    
    def save_predictions(self, predictions: pd.DataFrame):
        """Save model predictions to database"""
        self.insert_dataframe(predictions, 'predictions', if_exists='append')
    
    def save_alerts(self, alerts: pd.DataFrame):
        """Save alerts to database"""
        self.insert_dataframe(alerts, 'alerts', if_exists='append')
    
    def get_active_alerts(self) -> pd.DataFrame:
        """Get currently active alerts"""
        query = "SELECT * FROM alerts WHERE is_active = 1 ORDER BY alert_date DESC"
        return self.query_to_dataframe(query)
    
    def deactivate_alert(self, alert_id: int):
        """Deactivate an alert"""
        with self.engine.connect() as conn:
            conn.execute(f"UPDATE alerts SET is_active = 0 WHERE id = {alert_id}")
            conn.commit()
    
    def save_model_performance(self, performance_data: Dict):
        """Save model performance metrics"""
        df = pd.DataFrame([performance_data])
        self.insert_dataframe(df, 'model_performance', if_exists='append')
    
    def get_model_performance(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """Get model performance history"""
        query = "SELECT * FROM model_performance"
        if model_name:
            query += f" WHERE model_name = '{model_name}'"
        query += " ORDER BY evaluation_date DESC"
        return self.query_to_dataframe(query)
    
    def close(self):
        """Close database connection"""
        self.engine.dispose()

# Singleton instance
_db_manager = None

def get_db_manager() -> DatabaseManager:
    """Get singleton database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
