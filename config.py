# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 13:36:00 2025

@author: liangliang Wang
"""

from dataclasses import dataclass

@dataclass
class ModelConfig:
    n_estimators: int
    max_depth: int
    random_state: int
    test_size: float
    feature_selection_top_k: int
    min_samples_split: int
    min_samples_leaf: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    C_value: float
    alpha_value: float
    n_splits_cv: int
    perform_cv: bool

@dataclass
class AnalysisConfig:
    rat_file: str
    pig_file: str
    output_dir: str
    create_plots: bool
    save_models: bool