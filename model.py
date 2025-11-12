# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 13:36:00 2025

@author: liangliang Wang
"""

import numpy as np
import logging
import time
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.base import clone
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

class CrossSpeciesClassifier:
    def __init__(self, config):
        self.config = config
        self.models = []
        self.model_weights = []
        self.performance_history = []
        self.training_details = {}
        self.cv_results = {}
        self.base_models = self._initialize_models()
    
    def _initialize_models(self):
        return [
            ('SVC-RBF', SVC(kernel='rbf', C=self.config.C_value, gamma='scale', 
                           probability=True, random_state=self.config.random_state)),
            ('SVC-Linear', SVC(kernel='linear', C=self.config.C_value, 
                              probability=True, random_state=self.config.random_state)),
            ('RandomForest', RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                random_state=self.config.random_state,
                n_jobs=-1
            )),
            ('XGBoost', XGBClassifier(
                objective='multi:softprob',
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                reg_alpha=1.0,
                reg_lambda=1.0,
                random_state=self.config.random_state,
                n_jobs=-1
            )),
            ('LogisticRegression', LogisticRegression(
                C=self.config.C_value, 
                random_state=self.config.random_state, 
                max_iter=1000,
                multi_class='ovr',
                penalty='l2'
            )),
            ('RidgeClassifier', RidgeClassifier(
                alpha=self.config.alpha_value, 
                random_state=self.config.random_state
            ))
        ]
    
    def adaptive_feature_selection(self, X_source, X_target, y_source, top_k=None):
        if top_k is None:
            top_k = self.config.feature_selection_top_k
            
        n_features = X_source.shape[1]
        top_k = min(top_k, n_features)
        
        source_variances = np.var(X_source, axis=0)
        target_variances = np.var(X_target, axis=0)
        
        try:
            from sklearn.feature_selection import f_classif
            f_scores, _ = f_classif(X_source, y_source)
            f_scores = np.nan_to_num(f_scores, nan=0.0)
        except:
            f_scores = source_variances
        
        combined_scores = []
        for i in range(n_features):
            src_importance = f_scores[i] if i < len(f_scores) else source_variances[i]
            tgt_variance = target_variances[i] if i < len(target_variances) else 0
            score = src_importance * 0.6 + tgt_variance * 0.4
            combined_scores.append(score)
        
        valid_indices = list(range(n_features))
        top_indices = sorted(valid_indices, key=lambda i: combined_scores[i], reverse=True)[:top_k]
        
        feature_selection_details = {
            'total_features': n_features,
            'selected_features': top_k,
            'selection_method': 'F-test + Variance',
            'top_feature_indices': top_indices[:10]
        }
        
        logger.info(f"Selected {len(top_indices)} features from {n_features} total features")
        
        return X_source[:, top_indices], X_target[:, top_indices], top_indices, feature_selection_details
    
    def perform_cross_validation(self, X, y, n_splits=None):
        if n_splits is None:
            n_splits = self.config.n_splits_cv
            
        logger.info(f"Performing {n_splits}-fold cross-validation")
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.config.random_state)
        
        if hasattr(self, 'selected_features'):
            X_selected = X[:, self.selected_features]
        else:
            X_selected = X
        
        for name, model in self.models:
            try:
                cv_scores = cross_validate(
                    model, X_selected, y, 
                    cv=skf,
                    scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
                    return_train_score=True,
                    n_jobs=-1
                )
                
                cv_results[name] = {
                    'test_accuracy': cv_scores['test_accuracy'].tolist(),
                    'test_precision': cv_scores['test_precision_weighted'].tolist(),
                    'test_recall': cv_scores['test_recall_weighted'].tolist(),
                    'test_f1': cv_scores['test_f1_weighted'].tolist(),
                    'train_accuracy': cv_scores['train_accuracy'].tolist(),
                    'train_precision': cv_scores['train_precision_weighted'].tolist(),
                    'train_recall': cv_scores['train_recall_weighted'].tolist(),
                    'train_f1': cv_scores['train_f1_weighted'].tolist()
                }
                
                test_acc_mean = np.mean(cv_scores['test_accuracy'])
                train_acc_mean = np.mean(cv_scores['train_accuracy'])
                logger.info(f"  {name}: Train accuracy={train_acc_mean:.4f}, Test accuracy={test_acc_mean:.4f}")
                    
            except Exception as e:
                logger.warning(f"Model {name} cross-validation failed: {e}")
                cv_results[name] = {'error': str(e)}
                continue
        
        self.cv_results = cv_results
        return cv_results
    
    def fit(self, X_source, y_source, X_target, y_target):
        logger.info("Starting cross-species training")
        logger.info(f"Source data shape: {X_source.shape}, Target data shape: {X_target.shape}")
        
        start_time = time.time()
        
        self.training_details['config'] = {
            'source_samples': X_source.shape[0],
            'target_samples': X_target.shape[0],
            'features': X_source.shape[1],
            'source_classes': list(np.unique(y_source)),
            'target_classes': list(np.unique(y_target))
        }
        
        X_source_selected, X_target_selected, selected_indices, feature_selection_details = self.adaptive_feature_selection(
            X_source, X_target, y_source
        )
        
        self.selected_features = selected_indices
        self.training_details['feature_selection'] = feature_selection_details
        
        if len(X_target_selected) > 10:
            X_target_train, X_val, y_target_train, y_val = train_test_split(
                X_target_selected, y_target, test_size=self.config.test_size, 
                stratify=y_target, random_state=self.config.random_state
            )
        else:
            X_target_train, y_target_train = X_target_selected, y_target
            X_val, y_val = X_target_selected, y_target
        
        logger.info(f"Training set: {X_target_train.shape}, Validation set: {X_val.shape}")
        
        trained_models = []
        model_training_details = {}
        
        for name, base_model in self.base_models:
            logger.info(f"Training {name}")
            
            try:
                model_start_time = time.time()
                model_pretrained = clone(base_model)
                model_pretrained.fit(X_source_selected, y_source)
                training_time = time.time() - model_start_time
                
                trained_models.append((f"{name}_pretrained", model_pretrained))
                model_training_details[f"{name}_pretrained"] = {
                    'training_time_seconds': training_time,
                    'model_type': name
                }
                
                if len(X_target_train) > 5:
                    try:
                        model_start_time = time.time()
                        model_finetuned = clone(base_model)
                        X_combined = np.vstack([X_source_selected, X_target_train])
                        y_combined = np.concatenate([y_source, y_target_train])
                        model_finetuned.fit(X_combined, y_combined)
                        training_time = time.time() - model_start_time
                        
                        trained_models.append((f"{name}_finetuned", model_finetuned))
                        model_training_details[f"{name}_finetuned"] = {
                            'training_time_seconds': training_time,
                            'model_type': name,
                            'combined_samples': len(X_combined)
                        }
                    except Exception as e:
                        logger.warning(f"{name} fine-tuning failed: {e}")
                
            except Exception as e:
                logger.error(f"{name} training failed: {e}")
                continue
        
        if not trained_models:
            logger.warning("All models failed, using LogisticRegression as fallback")
            fallback_model = LogisticRegression(max_iter=1000, C=0.01)
            fallback_model.fit(X_source_selected, y_source)
            trained_models.append(('Fallback_Logistic', fallback_model))
        
        self.model_weights, model_performances = self._dynamic_ensemble_weighting(trained_models, X_val, y_val)
        self.models = trained_models
        
        self.training_details['model_training'] = model_training_details
        self.training_details['ensemble_weighting'] = model_performances
        
        if self.config.perform_cv:
            cv_results = self.perform_cross_validation(X_target, y_target)
            self.training_details['cross_validation'] = cv_results
        
        try:
            val_metrics = self.evaluate(X_val, y_val, already_selected=True)
            self.performance_history.append(val_metrics)
            self.training_details['validation_performance'] = val_metrics
        except Exception as e:
            logger.warning(f"Validation evaluation failed: {e}")
        
        total_training_time = time.time() - start_time
        self.training_details['total_training_time_seconds'] = total_training_time
        self.training_details['final_ensemble_size'] = len(self.models)
        
        logger.info(f"Training completed. Ensemble contains {len(self.models)} models")
        logger.info(f"Total training time: {total_training_time:.2f} seconds")
        
        return self
    
    def _dynamic_ensemble_weighting(self, models, X_val, y_val):
        weights = []
        model_performances = {}
        
        for name, model in models:
            try:
                start_time = time.time()
                y_pred = model.predict(X_val)
                inference_time = time.time() - start_time
                
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                
                weight = f1 + 1e-8
                weights.append(weight)
                
                model_performances[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'inference_time_seconds': inference_time,
                    'weight_assigned': weight
                }
                
                logger.info(f"  Model {name}: F1={f1:.4f}, Accuracy={accuracy:.4f}")
                
            except Exception as e:
                logger.warning(f"Model {name} evaluation failed: {e}")
                weights.append(1e-8)
                model_performances[name] = {
                    'error': str(e),
                    'weight_assigned': 1e-8
                }
        
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights] if total_weight > 0 else [1/len(weights)] * len(weights)
        
        for i, (name, _) in enumerate(models):
            if name in model_performances:
                model_performances[name]['normalized_weight'] = normalized_weights[i]
        
        return normalized_weights, model_performances
    
    def _select_features(self, X):
        if not hasattr(self, 'selected_features'):
            logger.warning("No feature selection information found, using all features")
            return X
        
        valid_indices = [idx for idx in self.selected_features if idx < X.shape[1]]
        if len(valid_indices) == 0:
            logger.warning("No valid feature indices found, using all features")
            return X
        
        return X[:, valid_indices]
    
    def predict(self, X, already_selected=False):
        if not self.models:
            raise ValueError("Model not trained. Please call fit() first.")
        
        if not already_selected:
            X_selected = self._select_features(X)
        else:
            X_selected = X
        
        predictions = []
        for (name, model), weight in zip(self.models, self.model_weights):
            try:
                pred = model.predict(X_selected)
                predictions.append((pred, weight, name))
            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {e}")
                continue
        
        if not predictions:
            raise ValueError("All model predictions failed")
        
        final_predictions = []
        for i in range(len(X_selected)):
            class_votes = {}
            for pred, weight, name in predictions:
                class_label = pred[i]
                if class_label not in class_votes:
                    class_votes[class_label] = 0
                class_votes[class_label] += weight
            
            final_predictions.append(max(class_votes.items(), key=lambda x: x[1])[0])
        
        return np.array(final_predictions)
    
    def predict_proba(self, X, already_selected=False):
        if not self.models:
            raise ValueError("Model not trained. Please call fit() first.")
        
        if not already_selected:
            X_selected = self._select_features(X)
        else:
            X_selected = X
        
        proba_sum = None
        total_weight = 0
        
        for (name, model), weight in zip(self.models, self.model_weights):
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_selected)
                    if proba_sum is None:
                        proba_sum = proba * weight
                    else:
                        if proba_sum.shape == proba.shape:
                            proba_sum += proba * weight
                        else:
                            logger.warning(f"Model {name} probability shape mismatch")
                            continue
                    total_weight += weight
            except Exception as e:
                logger.warning(f"Model {name} probability prediction failed: {e}")
                continue
        
        if proba_sum is None:
            n_classes = len(np.unique([model.predict(X_selected[:1])[0] for name, model in self.models]))
            return np.ones((len(X_selected), n_classes)) / n_classes
        
        return proba_sum / total_weight
    
    def evaluate(self, X, y, return_predictions=False, already_selected=False):
        start_time = time.time()
        y_pred = self.predict(X, already_selected=already_selected)
        y_proba = self.predict_proba(X, already_selected=already_selected)
        inference_time = time.time() - start_time
        
        metrics = {
            'Accuracy': accuracy_score(y, y_pred),
            'Precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'F1_Score': f1_score(y, y_pred, average='weighted', zero_division=0),
            'Confusion_Matrix': confusion_matrix(y, y_pred).tolist(),
            'Inference_Time_Seconds': inference_time,
            'Samples_Per_Second': len(X) / inference_time if inference_time > 0 else 0
        }
        
        try:
            class_report = classification_report(y, y_pred, output_dict=True, zero_division=0)
            metrics['Class_Report'] = class_report
        except Exception as e:
            logger.warning(f"Classification report generation failed: {e}")
        
        if y_proba is not None and len(np.unique(y)) > 1:
            try:
                if len(np.unique(y)) == 2:
                    metrics['ROC_AUC'] = roc_auc_score(y, y_proba[:, 1])
                else:
                    metrics['ROC_AUC'] = roc_auc_score(y, y_proba, multi_class='ovr', average='weighted')
            except Exception as e:
                logger.warning(f"ROC AUC calculation failed: {e}")
                metrics['ROC_AUC'] = 0.0
        
        if return_predictions:
            return metrics, y_pred, y_proba
        return metrics
    
    def get_training_summary(self):
        return {
            'training_details': self.training_details,
            'performance_history': self.performance_history,
            'cross_validation_results': self.cv_results,
            'final_ensemble': {
                'model_count': len(self.models),
                'model_names': [name for name, _ in self.models],
                'model_weights': self.model_weights
            }
        }
    
    def save_model(self, filepath):
        model_data = {
            'models': self.models,
            'model_weights': self.model_weights,
            'selected_features': getattr(self, 'selected_features', None),
            'config': self.config,
            'performance_history': self.performance_history,
            'training_details': self.training_details,
            'cv_results': self.cv_results
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath, config=None):
        model_data = joblib.load(filepath)
        
        if config is None:
            config = model_data['config']
        
        predictor = cls(config)
        predictor.models = model_data['models']
        predictor.model_weights = model_data['model_weights']
        predictor.selected_features = model_data['selected_features']
        predictor.performance_history = model_data['performance_history']
        predictor.training_details = model_data.get('training_details', {})
        predictor.cv_results = model_data.get('cv_results', {})
        
        logger.info(f"Model loaded from {filepath}")
        return predictor