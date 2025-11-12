# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 13:36:00 2025

@author: liangliang Wang
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def save_results(results, detailed_results, excel_data, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results_summary = {}
    for direction, result in results.items():
        if 'metrics' in result:
            serializable_metrics = {k: v for k, v in result['metrics'].items()}
            results_summary[direction] = serializable_metrics
        else:
            results_summary[direction] = {'error': result['error']}
    
    with open(output_path / 'results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    with open(output_path / 'detailed_results.json', 'w') as f:
        serializable_results = convert_to_serializable(detailed_results)
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    excel_data.to_excel(output_path / 'machine_learning_results.xlsx', index=False, engine='openpyxl')
    
    save_text_report(detailed_results, output_path / 'analysis_report.txt')
    
    logger.info(f"Results saved to {output_dir}")

def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if isinstance(k, (np.integer, np.int32, np.int64)):
                k = str(k)
            elif not isinstance(k, (str, int, float, bool)) and k is not None:
                k = str(k)
            new_dict[k] = convert_to_serializable(v)
        return new_dict
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def save_text_report(detailed_results, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Cross-Species Classification Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Execution Time: {detailed_results['execution_timestamp']}\n\n")
        
        f.write("Data Information:\n")
        data_info = detailed_results['data_info']
        f.write(f"- Rat Samples: {data_info['rat_samples']}\n")
        f.write(f"- Pig Samples: {data_info['pig_samples']}\n")
        f.write(f"- Features: {data_info['features']}\n")
        f.write(f"- Rat Classes: {data_info['rat_classes']}\n")
        f.write(f"- Pig Classes: {data_info['pig_classes']}\n")
        f.write(f"- Rat Class Distribution: {data_info['rat_class_distribution']}\n")
        f.write(f"- Pig Class Distribution: {data_info['pig_class_distribution']}\n\n")
        
        f.write("Model Configuration:\n")
        for key, value in detailed_results['training_config'].items():
            f.write(f"- {key}: {value}\n")
        f.write("\n")
        
        f.write("Cross-Species Prediction Results:\n")
        for direction, result in detailed_results['cross_species_results'].items():
            f.write(f"\n{direction}:\n")
            if 'error' in result:
                f.write(f"  Error: {result['error']}\n")
            else:
                metrics = result['performance_metrics']
                f.write(f"  Accuracy: {metrics['Accuracy']:.4f}\n")
                f.write(f"  F1-Score: {metrics['F1_Score']:.4f}\n")
                if 'ROC_AUC' in metrics:
                    f.write(f"  ROC AUC: {metrics['ROC_AUC']:.4f}\n")
                f.write(f"  Inference Time: {metrics['Inference_Time_Seconds']:.4f}s\n")
                f.write(f"  Samples/Second: {metrics['Samples_Per_Second']:.2f}\n")
                
                training_details = result['training_summary']['training_details']
                f.write(f"  Total Training Time: {training_details.get('total_training_time_seconds', 'N/A')}s\n")
                f.write(f"  Final Ensemble Size: {training_details.get('final_ensemble_size', 'N/A')}\n")
                
                if 'feature_selection' in training_details:
                    fs = training_details['feature_selection']
                    f.write(f"  Feature Selection: Selected {fs.get('selected_features', 'N/A')} from {fs.get('total_features', 'N/A')} features\n")
                
                cv_results = training_details.get('cross_validation', {})
                if cv_results:
                    f.write("  Cross-Validation Results:\n")
                    for model_name, scores in cv_results.items():
                        if 'error' not in scores:
                            test_acc = np.mean(scores['test_accuracy'])
                            train_acc = np.mean(scores['train_accuracy'])
                            f.write(f"    {model_name}: Train={train_acc:.4f}, Test={test_acc:.4f}\n")

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('cross_species_analysis.log'),
            logging.StreamHandler()
        ]
    )