"""
Main Pipeline Script
End-to-end crisis forecasting pipeline
"""
import argparse
import sys
import os
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.db_manager import DatabaseManager
from database.data_loader import load_data_to_database
from preprocessing.data_cleaner import DataCleaner
from preprocessing.temporal_aligner import TemporalAligner
from preprocessing.data_fusion import DataFusion
from preprocessing.feature_engineer import FeatureEngineer
from preprocessing.crisis_detector import CrisisLabeler
from models.ensemble_model import EnsembleModel
from models.evaluator import ModelEvaluator
from risk_assessment.risk_scorer import RiskScorer, AlertGenerator
from risk_assessment.scenario_simulator import ScenarioSimulator
from utils.logger import logger

def collect_data():
    """Step 1: Collect and load data"""
    logger.info("=" * 80)
    logger.info("STEP 1: DATA COLLECTION")
    logger.info("=" * 80)
    
    load_data_to_database()
    logger.info("✓ Data collection complete\n")

def preprocess_data():
    """Step 2: Preprocess and engineer features"""
    logger.info("=" * 80)
    logger.info("STEP 2: DATA PREPROCESSING")
    logger.info("=" * 80)
    
    # Load data
    db = DatabaseManager()
    datasets = {}
    
    for domain in ['climate', 'health', 'food', 'economic']:
        datasets[domain] = db.get_table_data(f'{domain}_data')
        logger.info(f"Loaded {domain} data: {len(datasets[domain])} records")
    
    # Clean data
    cleaner = DataCleaner()
    for domain, df in datasets.items():
        datasets[domain] = cleaner.clean_dataset(df, domain)
        cleaner.validate_data(df, domain)
    
    # Temporal alignment
    aligner = TemporalAligner()
    datasets = aligner.align_datasets(datasets)
    
    # Data fusion
    fusion = DataFusion()
    fused_df = fusion.fuse_data(datasets)
    fused_df = fusion.create_composite_indicators(fused_df)
    
    # Feature engineering
    engineer = FeatureEngineer()
    engineered_df = engineer.engineer_features(fused_df)
    
    # Crisis labeling
    labeler = CrisisLabeler()
    labeled_df = labeler.label_crises(engineered_df)
    
    # Save processed data
    engineered_df.to_csv('processed_data.csv', index=False)
    labeled_df.to_csv('labeled_data.csv', index=False)
    
    logger.info("✓ Data preprocessing complete\n")
    logger.info(f"Final dataset: {len(labeled_df)} records, {len(labeled_df.columns)} features")
    
    return labeled_df

def train_models(df):
    """Step 3: Train ML models"""
    logger.info("=" * 80)
    logger.info("STEP 3: MODEL TRAINING")
    logger.info("=" * 80)
    
    # Prepare training data
    feature_cols = [col for col in df.columns if col not in ['date', 'region', 'country'] 
                    and not col.startswith('crisis_')]
    
    X = df[feature_cols]
    y = df.get('crisis_composite', df.get('crisis_any', 0))
    
    # Split data (80-20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Train ensemble
    ensemble = EnsembleModel()
    ensemble.train(X_train, y_train)
    
   # Evaluate
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(ensemble, X_test, y_test)
    
    logger.info("\n" + evaluator.get_metrics_summary(metrics))
    
    # Save model
    ensemble.save_model('models/ensemble_model.pkl')
    logger.info("✓ Model training complete\n")
    
    return ensemble, X_test, y_test

def generate_predictions(model, df):
    """Step 4: Generate predictions and risk scores"""
    logger.info("=" * 80)
    logger.info("STEP 4: PREDICTION GENERATION")
    logger.info("=" * 80)
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['date', 'region', 'country'] 
                    and not col.startswith('crisis_')]
    
    X = df[feature_cols]
    
    # Generate predictions
    predictions = model.predict_proba(X)[:, 1]
    
    # Create predictions dataframe
    pred_df = pd.DataFrame({
        'date': df['date'],
        'region': df['region'],
        'probability': predictions,
        'composite_crisis_score': df.get('composite_crisis_score', 50)
    })
    
    # Calculate risk scores
    scorer = RiskScorer()
    risk_df = scorer.score_predictions(pred_df)
    
    # Generate alerts
    alert_gen = AlertGenerator()
    alerts_df = alert_gen.generate_alerts(risk_df)
    
    # Save predictions and alerts
    db = DatabaseManager()
    db.save_predictions(pred_df)
    db.save_alerts(alerts_df)
    
    risk_df.to_csv('predictions.csv', index=False)
    alerts_df.to_csv('alerts.csv', index=False)
    
    logger.info(f"✓ Generated {len(pred_df)} predictions")
    logger.info(f"✓ Generated {len(alerts_df)} alerts\n")
    
    return risk_df, alerts_df

def run_scenarios(model, df):
    """Step 5: Run scenario simulations"""
    logger.info("=" * 80)
    logger.info("STEP 5: SCENARIO SIMULATIONS")
    logger.info("=" * 80)
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['date', 'region', 'country'] 
                    and not col.startswith('crisis_')]
    
    X = df[feature_cols].tail(12)  # Last 12 months as basis
    
    # Run simulations
    simulator = ScenarioSimulator()
    scenarios = simulator.simulate_scenarios(model, X)
    
    # Compare scenarios
    comparison = simulator.compare_scenarios(scenarios)
    
    logger.info("\nScenario Comparison:")
    logger.info(comparison.to_string())
    
    logger.info("\n✓ Scenario simulations complete\n")
    
    return scenarios

def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description='Crisis Forecasting Pipeline')
    parser.add_argument('--collect-data', action='store_true', help='Collect and load data')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess data')
    parser.add_argument('--train-models', action='store_true', help='Train ML models')
    parser.add_argument('--predict', action='store_true', help='Generate predictions')
    parser.add_argument('--scenarios', action='store_true', help='Run scenario simulations')
    parser.add_argument('--all', action='store_true', help='Run complete pipeline')
    parser.add_argument('--dashboard', action='store_true', help='Launch Streamlit dashboard')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Execute pipeline
    start_time = datetime.now()
    logger.info("Starting Crisis Forecasting Pipeline")
    logger.info(f"Start time: {start_time}")
    logger.info("=" * 80 + "\n")
    
    try:
        if args.all or args.collect_data:
            collect_data()
        
        if args.all or args.preprocess:
            processed_df = preprocess_data()
        else:
            # Load existing processed data
            if os.path.exists('labeled_data.csv'):
                processed_df = pd.read_csv('labeled_data.csv')
            else:
                logger.error("No processed data found. Run with --preprocess first.")
                return
        
        if args.all or args.train_models:
            model, X_test, y_test = train_models(processed_df)
        else:
            # Load existing model
            if os.path.exists('models/ensemble_model.pkl'):
                model = EnsembleModel()
                model.load_model('models/ensemble_model.pkl')
            else:
                logger.error("No trained model found. Run with --train-models first.")
                return
        
        if args.all or args.predict:
            risk_df, alerts_df = generate_predictions(model, processed_df)
        
        if args.all or args.scenarios:
            scenarios = run_scenarios(model, processed_df)
        
        if args.dashboard:
            logger.info("Launching Streamlit dashboard...")
            import subprocess
            subprocess.run(['streamlit', 'run', 'dashboard/app.py'])
        
        # Summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"End time: {end_time}")
        logger.info(f"Duration: {duration}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
