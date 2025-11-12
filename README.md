**Cross-Species Time Prediction Framework—xSpecML: User Manual**

**1\. Overview**

The **xSpecML** is a comprehensive machine learning pipeline designed for cross-species prediction of time-series transcriptomic data. This framework enables researchers to build robust classification models that can transfer knowledge between different species (e.g., rat to pig or vice versa) for predicting time-dependent biological states.

**Key Features**

*   **Ensemble Learning**: Combines multiple machine learning algorithms with dynamic weighting
*   **Cross-Species Transfer**: Enables model training on one species and prediction on another
*   **Adaptive Feature Selection**: Automatically selects biologically relevant features
*   **Comprehensive Evaluation**: Provides extensive performance metrics and visualizations
*   **Reproducible Analysis**: Configurable parameters for consistent scientific results

**2\. System Requirements**

**Software Dependencies**

Python >= 3.8

pandas >= 1.3.0

numpy >= 1.21.0

scikit-learn >= 1.0.0

matplotlib >= 3.5.0

seaborn >= 0.11.0

xgboost >= 1.5.0

openpyxl >= 3.0.0

joblib >= 1.1.0

**Hardware Recommendations**

*   **RAM**: Minimum 16GB, 32GB recommended for large datasets
*   **Storage**: 1GB free space for results and models
*   **CPU**: Multi-core processor for efficient cross-validation

**3\. Installation**

**Method 1: Using pip**

pip install pandas numpy scikit-learn matplotlib seaborn xgboost openpyxl joblib

**Method 2: Using conda**

conda install -c conda-forge pandas numpy scikit-learn matplotlib seaborn xgboost openpyxl joblib

**4\. Data Preparation**

**Input File Format**

The framework requires two Excel files (rat and pig data) with the following structure:

**Data Sheet (sheet\_name='data')**

*   **Rows**: Samples (biological replicates)
*   **Columns**: Features (gene expression values)
*   **Index**: Sample identifiers

Example structure:

| SampleID | Gene1 | Gene2 | ... | GeneN |
| --- | --- | --- | --- | --- |
| S1_Rat | 5.23 | 8.45 | ... | 6.78 |
| S2_Rat | 4.89 | 7.92 | ... | 5.43 |

**Target Sheet (sheet\_name='target')**

*   **Rows**: Samples (must match data sheet)
*   **Columns**: Class labels
*   **Index**: Sample identifiers

Example structure:

| SampleID | class |
| --- | --- |
| S1_Rat | C |
| S2_Rat | 4h |

**Class Labels**

The framework expects time-series labels in the order: \['C', '4h', '8h', '12h', '16h', '20h', '24h', '48h'\]. Please modify according to the actual situation.

**5\. Configuration**

**Model Configuration (config.py)**

python

@dataclass

class ModelConfig:

n\_estimators: int = 50 _\# Number of trees for ensemble methods_

max\_depth: int = 5 _\# Maximum tree depth_

random\_state: int = 42 _\# Random seed for reproducibility_

test\_size: float = 0.3 _\# Validation set proportion_

feature\_selection\_top\_k: int = 50 _\# Number of features to select_

min\_samples\_split: int = 10 _\# Minimum samples for node splitting_

min\_samples\_leaf: int = 5 _\# Minimum samples per leaf_

learning\_rate: float = 0.05 _\# Learning rate for XGBoost_

subsample: float = 0.7 _\# Subsample ratio for XGBoost_

colsample\_bytree: float = 0.7 _\# Feature subsample ratio for XGBoost_

C\_value: float = 0.1 _\# Regularization parameter for SVM_

alpha\_value: float = 10.0 _\# Regularization for Ridge classifier_

n\_splits\_cv: int = 5 _\# Cross-validation folds_

perform\_cv: bool = True _\# Enable cross-validation_

**Analysis Configuration**

@dataclass

class AnalysisConfig:

rat\_file: str = 'rat.xlsx'

pig\_file: str = 'pig.xlsx'

output\_dir: str = 'cross\_species\_results'

create\_plots: bool = True _\# Generate visualization plots_

save\_models: bool = True _\# Save trained models_

**6\. Usage**

**Basic Execution**

python main.py

**Custom Execution**

Modify the configuration in main.py:

python

def get\_model\_config():

return ModelConfig(

n\_estimators=100, _\# Increase for better performance_

max\_depth=7, _\# Deeper trees for complex patterns_

feature\_selection\_top\_k=100, _\# More features_

perform\_cv=True _\# Enable cross-validation_

)

def get\_analysis\_config():

return AnalysisConfig(

rat\_file='your\_rat\_data.xlsx',

pig\_file='your\_pig\_data.xlsx',

output\_dir='custom\_results',

create\_plots=True,

save\_models=True

)

**7\. Algorithm Details**

**Ensemble Methods**

The framework employs six base classifiers:

1.  **Support Vector Classifier (RBF kernel)**
2.  **Support Vector Classifier (Linear kernel)**
3.  **Random Forest**
4.  **XGBoost**
5.  **Logistic Regression**
6.  **Ridge Classifier**

**Feature Selection**

*   **Method**: Combined F-test and variance analysis
*   **Scoring**: 60% feature importance + 40% target variance
*   **Output**: Top-k most discriminative features

**Training Strategy**

1.  **Pre-training**: Models trained on source species data
2.  **Fine-tuning**: Selected models fine-tuned on combined source + target data
3.  **Ensemble Weighting**: Dynamic weighting based on validation performance

**8\. Output Files**

**Results Directory Structure**

text

cross\_species\_results/

├── results\_summary.json # Summary metrics

├── detailed\_results.json # Comprehensive results

├── analysis\_report.txt # Text summary

├── machine\_learning\_results.xlsx # Excel results

├── Rat2Pig\_model.joblib # Rat→Pig trained model

├── Pig2Rat\_model.joblib # Pig→Rat trained model

├── Rat2Pig\_cv\_results.png # Cross-validation plot

├── Rat2Pig\_roc.png # ROC curves

├── Rat2Pig\_cm.png # Confusion matrix

└── cross\_species\_analysis.log # Execution log

**Key Output Metrics**

*   **Accuracy**: Overall classification accuracy
*   **Precision**: Weighted precision across classes
*   **Recall**: Weighted recall across classes
*   **F1-Score**: Harmonic mean of precision and recall
*   **ROC-AUC**: Area under ROC curve (multi-class)
*   **Confusion Matrix**: Class-wise performance
*   **Cross-validation Scores**: Training and test performance

**9\. Interpretation of Results**

**Performance Metrics**

*   **Accuracy > 0.8**: Excellent cross-species transfer
*   **Accuracy 0.6-0.8**: Good predictive performance
*   **Accuracy < 0.6**: Limited cross-species applicability

**Feature Selection**

*   **High-variance features**: Conserved biological mechanisms
*   **F-test significant features**: Species-specific responses
*   **Selected feature count**: Optimal feature set size

**Model Validation**

*   **Cross-validation consistency**: Model reliability
*   **Training-test gap**: Overfitting assessment
*   **Ensemble diversity**: Complementary model strengths

**10\. Advanced Usage**

**Custom Data Integration**

python

_\# For different species or time points_

class\_order = \['Control', '1h', '2h', '4h', '8h', '12h'\] _\# Custom time points_

**Model Extension**

python

_\# Add custom models to the ensemble_

def \_initialize\_models(self):

base\_models = \[

_\# Existing models..._

('CustomModel', YourCustomClassifier(parameters))

\]

return base\_models

**Feature Selection Modification**

python

def adaptive\_feature\_selection(self, X\_source, X\_target, y\_source, top\_k=None):

_\# Implement custom feature selection criteria_

custom\_scores = your\_custom\_method(X\_source, X\_target, y\_source)

_\# Replace combined\_scores calculation_

**11\. Troubleshooting**

**Common Issues**

1.  **File Not Found Error**
    
        *   Verify file paths in AnalysisConfig
        *   Check Excel file permissions
2.  **Memory Errors**
    
        *   Reduce feature\_selection\_top\_k
        *   Decrease n\_estimators
        *   Use data sampling
3.  **Poor Performance**
    
        *   Increase feature\_selection\_top\_k
        *   Adjust max\_depth and n\_estimators
        *   Verify data quality and preprocessing
4.  **Convergence Warnings**
    
        *   Increase max\_iter for Logistic Regression
        *   Adjust learning\_rate for XGBoost

**Debug Mode**

Enable detailed logging by modifying the logging level in utils.py:

python

setup\_logging(level=logging.DEBUG)

**12\. Citation**

When using this framework in publications, please cite:

**13\. Support**

For technical support or questions:

*   Check the log file: cross\_species\_analysis.log
*   Verify input data format requirements
*   Ensure all dependencies are correctly installed
*   Contact:
*   Liangliang Wang
*   [Liangliang.wang@sxmu.edu.cn](mailto:Liangliang.wang@sxmu.edu.cn)
*   Shanxi Medical University

**14\. License**

This framework is available for academic use. For commercial applications, please contact the authors.

\*This user manual corresponds to version 1.0 of the Cross-Species Transcriptomic Classification Framework. For updates and additional resources, please contact the author.
