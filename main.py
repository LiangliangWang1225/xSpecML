# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 13:36:00 2025

@author: liangliang Wang
"""

#pip install -r requirements.txt

import logging
import pandas as pd
import numpy as np

from config import ModelConfig, AnalysisConfig
from data_processor import DataProcessor, load_and_preprocess_data
from model import CrossSpeciesClassifier
from visualizer import ResultVisualizer
from utils import save_results, setup_logging

logger = logging.getLogger(__name__)

def get_model_config():
    return ModelConfig(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        test_size=0.3,
        feature_selection_top_k=50,
        min_samples_split=10,
        min_samples_leaf=5,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        C_value=0.1,
        alpha_value=10.0,
        n_splits_cv=5,
        perform_cv=True
    )

def get_analysis_config():
    return AnalysisConfig(
        rat_file='rat.xlsx',
        pig_file='pig.xlsx',
        output_dir='cross_species_results',
        create_plots=True,
        save_models=True
    )

def run_cross_species_analysis(config, model_config):
    logger.info("Starting cross-species classification analysis")
    
    data_processor = DataProcessor(model_config)
    visualizer = ResultVisualizer()
    
    logger.info("Loading and preprocessing data...")
    (X_rat, y_rat_encoded, X_pig, y_pig_encoded, 
     data_info, processor) = load_and_preprocess_data(config.rat_file, config.pig_file, data_processor)
    
    logger.info(f"Processed data - Rat: {X_rat.shape}, Pig: {X_pig.shape}")
    
    rat_class_names = processor.get_class_names("rat")
    pig_class_names = processor.get_class_names("pig")
    ordered_class_names = processor.get_ordered_class_names()
    
    logger.info(f"Rat classes: {rat_class_names}")
    logger.info(f"Pig classes: {pig_class_names}")
    
    results = {}
    detailed_results = {
        'data_info': data_info,
        'training_config': model_config.__dict__,
        'execution_timestamp': pd.Timestamp.now().isoformat(),
        'cross_species_results': {}
    }
    
    directions = [
        ('Rat_to_Pig', X_rat, y_rat_encoded, X_pig, y_pig_encoded, rat_class_names, pig_class_names),
        ('Pig_to_Rat', X_pig, y_pig_encoded, X_rat, y_rat_encoded, pig_class_names, rat_class_names)
    ]
    
    for direction_name, X_source, y_source, X_target, y_target, source_class_names, target_class_names in directions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training: {direction_name}")
        logger.info(f"{'='*60}")
        
        try:
            classifier = CrossSpeciesClassifier(model_config)
            classifier.fit(X_source, y_source, X_target, y_target)
            
            metrics, predictions, probabilities = classifier.evaluate(X_target, y_target, return_predictions=True)
            
            predictions_decoded = processor.decode_labels(predictions, "pig" if direction_name == "Rat_to_Pig" else "rat")
            y_target_decoded = processor.decode_labels(y_target, "pig" if direction_name == "Rat_to_Pig" else "rat")
            
            training_summary = classifier.get_training_summary()
            
            results[direction_name] = {
                'metrics': metrics,
                'predictions': predictions,
                'predictions_decoded': predictions_decoded,
                'probabilities': probabilities,
                'model': classifier,
                'target_class_names': target_class_names
            }
            
            detailed_results['cross_species_results'][direction_name] = {
                'performance_metrics': metrics,
                'training_summary': training_summary,
                'target_class_names': target_class_names,
                'source_class_names': source_class_names
            }
            
            file_suffix = direction_name.replace('_to_', ' to ')
            
            if config.create_plots:
                try:
                    cv_results = training_summary['training_details'].get('cross_validation', {})
                    visualizer.plot_cross_validation_results(
                        cv_results,
                        f'Cross-Validation Results - {direction_name}',
                        f'{config.output_dir}/{file_suffix}_cv_results.png'
                    )
                except Exception as e:
                    logger.warning(f"Failed to plot cross-validation results: {e}")
                
                if probabilities is not None:
                    try:
                        visualizer.plot_roc_curves(
                            y_target, probabilities, ordered_class_names,
                            f'ROC Curves - {direction_name}',
                            f'{config.output_dir}/{file_suffix}_roc.png'
                        )
                    except Exception as e:
                        logger.warning(f"Failed to plot ROC curves: {e}")
                
                try:
                    visualizer.plot_confusion_matrix(
                        y_target, predictions, ordered_class_names,
                        f'Confusion Matrix - {direction_name}',
                        f'{config.output_dir}/{file_suffix}_cm.png'
                    )
                except Exception as e:
                    logger.warning(f"Failed to plot confusion matrix: {e}")
            
            if config.save_models:
                classifier.save_model(f'{config.output_dir}/{file_suffix}_model.joblib')
            
            logger.info(f"{direction_name} completed successfully")
            for metric, value in metrics.items():
                if metric not in ['Confusion_Matrix', 'Class_Report']:
                    logger.info(f"  {metric}: {value:.4f}")
                    
        except Exception as e:
            logger.error(f"{direction_name} failed: {e}")
            results[direction_name] = {'error': str(e)}
            detailed_results['cross_species_results'][direction_name] = {'error': str(e)}
    
    excel_data = prepare_excel_data(detailed_results)
    save_results(results, detailed_results, excel_data, config.output_dir)
    
    logger.info(f"\nAnalysis completed. Results saved to {config.output_dir}/")
    logger.info("Results Summary:")
    for direction, result in results.items():
        if 'metrics' in result:
            metrics = result['metrics']
            logger.info(f"  {direction}: Accuracy = {metrics['Accuracy']:.4f}, F1 = {metrics['F1_Score']:.4f}")
        elif 'error' in result:
            logger.info(f"  {direction}: Error = {result['error']}")
    
    return results, detailed_results, excel_data

def prepare_excel_data(detailed_results):
    excel_data = []
    
    for direction, result in detailed_results['cross_species_results'].items():
        if 'error' in result:
            excel_data.append({
                'Direction': direction,
                'Status': 'Failed',
                'Error_Message': result['error']
            })
        else:
            metrics = result['performance_metrics']
            training_details = result['training_summary']['training_details']
            feature_selection = training_details.get('feature_selection', {})
            
            cv_scores = training_details.get('cross_validation', {})
            cv_test_acc = []
            cv_train_acc = []
            
            for model_scores in cv_scores.values():
                if 'test_accuracy' in model_scores:
                    cv_test_acc.extend(model_scores['test_accuracy'])
                if 'train_accuracy' in model_scores:
                    cv_train_acc.extend(model_scores['train_accuracy'])
            
            cv_test_mean = np.mean(cv_test_acc) if cv_test_acc else 'N/A'
            cv_train_mean = np.mean(cv_train_acc) if cv_train_acc else 'N/A'
            
            excel_data.append({
                'Direction': direction,
                'Status': 'Success',
                'Accuracy': metrics['Accuracy'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1_Score': metrics['F1_Score'],
                'ROC_AUC': metrics.get('ROC_AUC', 'N/A'),
                'Inference_Time_Seconds': metrics['Inference_Time_Seconds'],
                'Samples_Per_Second': metrics['Samples_Per_Second'],
                'Total_Training_Time_Seconds': training_details.get('total_training_time_seconds', 'N/A'),
                'Ensemble_Model_Count': training_details.get('final_ensemble_size', 'N/A'),
                'Total_Features': feature_selection.get('total_features', 'N/A'),
                'Selected_Features': feature_selection.get('selected_features', 'N/A'),
                'Feature_Selection_Method': feature_selection.get('selection_method', 'N/A'),
                'CV_Training_Accuracy': cv_train_mean,
                'CV_Test_Accuracy': cv_test_mean,
                'Source_Samples': training_details.get('config', {}).get('source_samples', 'N/A'),
                'Target_Samples': training_details.get('config', {}).get('target_samples', 'N/A')
            })
    
    return pd.DataFrame(excel_data)

if __name__ == "__main__":
    setup_logging()
    
    model_config = get_model_config()
    analysis_config = get_analysis_config()
    
    try:
        results, detailed_results, excel_results = run_cross_species_analysis(analysis_config, model_config)
        
        print("\n" + "="*60)
        print("Cross-Species Classification Analysis Completed Successfully!")
        print("="*60)
        
        for direction, result in results.items():
            if 'metrics' in result:
                metrics = result['metrics']
                print(f"\n{direction}:")
                print(f"  Accuracy: {metrics['Accuracy']:.4f}")
                print(f"  F1-Score: {metrics['F1_Score']:.4f}")
                if 'ROC_AUC' in metrics:
                    print(f"  ROC AUC: {metrics['ROC_AUC']:.4f}")
                print(f"  Inference Time: {metrics['Inference_Time_Seconds']:.4f}s")
                
                if direction in detailed_results['cross_species_results']:
                    cv_info = detailed_results['cross_species_results'][direction]['training_summary']['training_details'].get('cross_validation', {})
                    if cv_info:
                        print("  Cross-Validation Results:")
                        for model_name, scores in cv_info.items():
                            if 'error' not in scores:
                                test_acc = np.mean(scores['test_accuracy'])
                                train_acc = np.mean(scores['train_accuracy'])
                                print(f"    {model_name}: Train={train_acc:.4f}, Test={test_acc:.4f}")
            else:
                print(f"\n{direction}: Error - {result['error']}")
        
        print(f"\nDetailed results saved to {analysis_config.output_dir}/ directory")
        print(f"- Results Summary: {analysis_config.output_dir}/results_summary.json")
        print(f"- Detailed Results: {analysis_config.output_dir}/detailed_results.json")
        print(f"- Analysis Report: {analysis_config.output_dir}/analysis_report.txt")
        print(f"- Excel Results: {analysis_config.output_dir}/machine_learning_results.xlsx")
        print(f"- Model Files: {analysis_config.output_dir}/*.joblib")
        print(f"- Visualization Plots: {analysis_config.output_dir}/*.png")
        
        print(f"\nExcel Results Preview:")
        print(excel_results.to_string(index=False))
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"Error: {e}")
        print("Please check file paths and data format.")