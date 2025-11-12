# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 13:36:00 2025

@author: liangliang Wang
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import logging

logger = logging.getLogger(__name__)

class ResultVisualizer:
    @staticmethod
    def _set_sci_style():
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 14,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'axes.linewidth': 1.5,
            'lines.linewidth': 2,
            'legend.fontsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
    
    @staticmethod
    def plot_cross_validation_results(cv_results, title, filename):
        ResultVisualizer._set_sci_style()
        
        models = list(cv_results.keys())
        if not models:
            return
        
        train_scores = []
        test_scores = []
        model_names = []
        
        for model_name, scores in cv_results.items():
            if 'error' in scores:
                continue
                
            train_mean = np.mean(scores['train_accuracy'])
            test_mean = np.mean(scores['test_accuracy'])
            train_scores.append(train_mean)
            test_scores.append(test_mean)
            model_names.append(model_name)
        
        if not train_scores:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, train_scores, width, label='Training Accuracy', 
                      color='#2E86AB', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, test_scores, width, label='Test Accuracy', 
                      color='#A23B72', alpha=0.8, edgecolor='black')
        
        for i, (train_score, test_score) in enumerate(zip(train_scores, test_scores)):
            ax.text(i - width/2, train_score + 0.01, f'{train_score:.3f}', 
                   ha='center', va='bottom', fontweight='bold')
            ax.text(i + width/2, test_score + 0.01, f'{test_score:.3f}', 
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        except Exception as e:
            logger.error(f"Failed to save plot {filename}: {e}")
        plt.close()
        
        plt.rcParams.update(plt.rcParamsDefault)
    
    @staticmethod
    def plot_roc_curves(y_true, y_proba, class_names, title, filename):
        ResultVisualizer._set_sci_style()
        
        n_classes = len(class_names)
        fig, ax = plt.subplots(figsize=(6, 6))
        
        sci_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
        
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=sci_colors[0], lw=3, 
                   label=f'ROC curve (AUC = {roc_auc:.3f})', alpha=0.8)
            ax.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', alpha=0.5)
            
        else:
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                color = sci_colors[i % len(sci_colors)]
                ax.plot(fpr, tpr, color=color, lw=3,
                       label=f'{class_names[i]} (AUC = {roc_auc:.3f})', alpha=0.8)
            
            ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        plt.tight_layout()
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        except Exception as e:
            logger.error(f"Failed to save plot {filename}: {e}")
        plt.close()
        
        plt.rcParams.update(plt.rcParamsDefault)
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names, title, filename, normalize=True):
        ResultVisualizer._set_sci_style()
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.round(cm * 100, 1)
            fmt = '.1f'
            cmap = 'Blues'
        else:
            fmt = 'd'
            cmap = 'Blues'
        
        fig, ax = plt.subplots(figsize=(7, 7))
        
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap, aspect='equal')
        
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=12)
        if normalize:
            cbar.set_label('Percentage (%)', rotation=270, labelpad=20)
        else:
            cbar.set_label('Count', rotation=270, labelpad=20)
        
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(title)
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text_color = "white" if cm[i, j] > thresh else "black"
                if normalize:
                    text = f'{cm[i, j]}%'
                else:
                    text = f'{cm[i, j]}'
                ax.text(j, i, text, ha="center", va="center",
                       color=text_color, fontweight='bold', fontsize=12)
        
        ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
        ax.tick_params(which="minor", bottom=False, left=False)
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        plt.tight_layout()
        
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        except Exception as e:
            logger.error(f"Failed to save plot {filename}: {e}")
        plt.close()
        
        plt.rcParams.update(plt.rcParamsDefault)