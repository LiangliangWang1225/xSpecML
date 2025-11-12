# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 13:36:00 2025

@author: liangliang Wang
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder
import logging
from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.label_encoders = {}
        self.class_order = ['C', '4h', '8h', '12h', '16h', '20h', '24h', '48h']
    
    def robust_scale(self, X: np.ndarray, feature_names: List[str] = None) -> np.ndarray:
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        X_scaled = X.copy()
        for i, feature in enumerate(feature_names):
            if feature not in self.scalers:
                self.scalers[feature] = RobustScaler()
            
            if len(np.unique(X[:, i])) > 1:
                X_scaled[:, i] = self.scalers[feature].fit_transform(X[:, i].reshape(-1, 1)).flatten()
            else:
                X_scaled[:, i] = 0.0
        
        return X_scaled
    
    def encode_labels(self, y: np.ndarray, label_name: str = "default") -> np.ndarray:
        if label_name not in self.label_encoders:
            unique_classes = np.unique(y)
            ordered_classes = [cls for cls in self.class_order if cls in unique_classes]
            remaining_classes = [cls for cls in unique_classes if cls not in ordered_classes]
            ordered_classes.extend(remaining_classes)
            
            self.label_encoders[label_name] = LabelEncoder()
            self.label_encoders[label_name].fit(ordered_classes)
        
        return self.label_encoders[label_name].transform(y)
    
    def decode_labels(self, y_encoded: np.ndarray, label_name: str = "default") -> np.ndarray:
        if label_name in self.label_encoders:
            return self.label_encoders[label_name].inverse_transform(y_encoded)
        return y_encoded
    
    def get_class_names(self, label_name: str = "default") -> List[str]:
        if label_name in self.label_encoders:
            return list(self.label_encoders[label_name].classes_)
        return []
    
    def get_ordered_class_names(self) -> List[str]:
        return self.class_order

def load_and_preprocess_data(rat_file: str, pig_file: str, processor: DataProcessor) -> Tuple:
    try:
        df_rat_data = pd.read_excel(rat_file, sheet_name='data', index_col=0)
        df_rat_target = pd.read_excel(rat_file, sheet_name='target', index_col=0)
        
        df_pig_data = pd.read_excel(pig_file, sheet_name='data', index_col=0)
        df_pig_target = pd.read_excel(pig_file, sheet_name='target', index_col=0)
        
        logger.info(f"Original data - Rat: {df_rat_data.shape}, Pig: {df_pig_data.shape}")
        
        common_rat_samples = df_rat_data.index.intersection(df_rat_target.index)
        common_pig_samples = df_pig_data.index.intersection(df_pig_target.index)
        
        df_rat_data = df_rat_data.loc[common_rat_samples]
        df_rat_target = df_rat_target.loc[common_rat_samples]
        df_pig_data = df_pig_data.loc[common_pig_samples]
        df_pig_target = df_pig_target.loc[common_pig_samples]
        
        logger.info(f"Aligned data - Rat: {df_rat_data.shape}, Pig: {df_pig_data.shape}")
        
        X_rat = df_rat_data.values
        y_rat = df_rat_target['class'].values
        X_pig = df_pig_data.values
        y_pig = df_pig_target['class'].values
        
        y_rat_encoded = processor.encode_labels(y_rat, "rat")
        y_pig_encoded = processor.encode_labels(y_pig, "pig")
        
        rat_class_names = processor.get_class_names("rat")
        pig_class_names = processor.get_class_names("pig")
        
        logger.info(f"Rat classes: {rat_class_names}")
        logger.info(f"Pig classes: {pig_class_names}")
        
        feature_names = df_rat_data.columns.tolist()
        X_rat_processed = processor.robust_scale(X_rat, feature_names)
        X_pig_processed = processor.robust_scale(X_pig, feature_names)
        
        data_info = {
            'rat_samples': len(X_rat),
            'pig_samples': len(X_pig),
            'features': len(feature_names),
            'rat_classes': rat_class_names,
            'pig_classes': pig_class_names,
            'rat_class_distribution': dict(zip(*np.unique(y_rat, return_counts=True))),
            'pig_class_distribution': dict(zip(*np.unique(y_pig, return_counts=True))),
            'feature_names': feature_names
        }
        
        return (X_rat_processed, y_rat_encoded, X_pig_processed, y_pig_encoded, 
                data_info, processor)
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise