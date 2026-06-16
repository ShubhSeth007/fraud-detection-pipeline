import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def evaluate_production_drift(reference_path: str, current_path: str, output_html_path: str):
    print("--- Loading Datasets for Drift Verification ---")
    reference_data = pd.read_csv(reference_path)
    current_data = pd.read_csv(current_path)
    
    shared_features = [col for col in reference_data.columns if col in current_data.columns and col != 'isFraud']
    
    print("--- Generating Evidently AI Data Drift Profile ---")
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(
        reference_data=reference_data[shared_features], 
        current_data=current_data[shared_features]
    )
    
    drift_report.save_html(output_html_path)
    print(f"Drift diagnostic suite compiled successfully! Result saved to: {output_html_path}")

if __name__ == "__main__":
    # Generate mock tracking data to evaluate operational validity
    df_ref = pd.DataFrame(np.random.randn(1000, 3), columns=['TransactionAmt', 'card1', 'dist1'])
    df_cur = pd.DataFrame(np.random.randn(1000, 3) + 0.6, columns=['TransactionAmt', 'card1', 'dist1']) 
    
    df_ref.to_csv("train_baseline.csv", index=False)
    df_cur.to_csv("production_logs.csv", index=False)
    
    evaluate_production_drift(
        reference_path="train_baseline.csv", 
        current_path="production_logs.csv", 
        output_html_path="data_drift_report.html"
    )
