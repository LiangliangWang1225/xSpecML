# xSpecML
The xSpecML is a comprehensive machine learning pipeline designed for cross-species prediction of time-series transcriptomic data. This framework enables researchers to build robust classification models that can transfer knowledge between different species (e.g., rat to pig or vice versa) for predicting time-dependent biological states.
Key Features
•	Ensemble Learning: Combines multiple machine learning algorithms with dynamic weighting
•	Cross-Species Transfer: Enables model training on one species and prediction on another
•	Adaptive Feature Selection: Automatically selects biologically relevant features
•	Comprehensive Evaluation: Provides extensive performance metrics and visualizations
•	Reproducible Analysis: Configurable parameters for consistent scientific results
2. System Requirements
Software Dependencies
Python >= 3.8
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
xgboost >= 1.5.0
openpyxl >= 3.0.0
joblib >= 1.1.0
Hardware Recommendations
•	RAM: Minimum 16GB, 32GB recommended for large datasets
•	Storage: 1GB free space for results and models
•	CPU: Multi-core processor for efficient cross-validation
3. Installation
Method 1: Using pip
pip install pandas numpy scikit-learn matplotlib seaborn xgboost openpyxl joblib
Method 2: Using conda
conda install -c conda-forge pandas numpy scikit-learn matplotlib seaborn xgboost openpyxl joblib
4. Data Preparation
Input File Format
The framework requires two Excel files (rat and pig data) with the following structure:
Data Sheet (sheet_name='data')
•	Rows: Samples (biological replicates)
•	Columns: Features (gene expression values)
•	Index: Sample identifiers
Example structure:
SampleID	Gene1	Gene2	...	GeneN
S1_Rat	5.23	8.45	...	6.78
S2_Rat	4.89	7.92	...	5.43
Target Sheet (sheet_name='target')
•	Rows: Samples (must match data sheet)
•	Columns: Class labels
•	Index: Sample identifiers
Example structure:
SampleID	class
S1_Rat	C
S2_Rat	4h
Class Labels
The framework expects time-series labels in the order: ['C', '4h', '8h', '12h', '16h', '20h', '24h', '48h']. Please modify according to the actual situation.
5. Configuration
Model Configuration (config.py)
python
@dataclass
class ModelConfig:
    n_estimators: int = 50        # Number of trees for ensemble methods
    max_depth: int = 5            # Maximum tree depth
    random_state: int = 42        # Random seed for reproducibility
    test_size: float = 0.3        # Validation set proportion
    feature_selection_top_k: int = 50  # Number of features to select
    min_samples_split: int = 10   # Minimum samples for node splitting
    min_samples_leaf: int = 5     # Minimum samples per leaf
    learning_rate: float = 0.05   # Learning rate for XGBoost
    subsample: float = 0.7        # Subsample ratio for XGBoost
    colsample_bytree: float = 0.7 # Feature subsample ratio for XGBoost
    C_value: float = 0.1          # Regularization parameter for SVM
    alpha_value: float = 10.0     # Regularization for Ridge classifier
    n_splits_cv: int = 5          # Cross-validation folds
    perform_cv: bool = True       # Enable cross-validation
Analysis Configuration
@dataclass
class AnalysisConfig:
    rat_file: str = 'rat.xlsx'
    pig_file: str = 'pig.xlsx'
    output_dir: str = 'cross_species_results'
    create_plots: bool = True      # Generate visualization plots
    save_models: bool = True       # Save trained models
6. Usage
Basic Execution
python main.py
Custom Execution
Modify the configuration in main.py:
python
def get_model_config():
    return ModelConfig(
        n_estimators=100,      # Increase for better performance
        max_depth=7,           # Deeper trees for complex patterns
        feature_selection_top_k=100,  # More features
        perform_cv=True        # Enable cross-validation
    )

def get_analysis_config():
    return AnalysisConfig(
        rat_file='your_rat_data.xlsx',
        pig_file='your_pig_data.xlsx',
        output_dir='custom_results',
        create_plots=True,
        save_models=True
    )
7. Algorithm Details
Ensemble Methods
The framework employs six base classifiers:
1.	Support Vector Classifier (RBF kernel)
2.	Support Vector Classifier (Linear kernel)
3.	Random Forest
4.	XGBoost
5.	Logistic Regression
6.	Ridge Classifier
Feature Selection
•	Method: Combined F-test and variance analysis
•	Scoring: 60% feature importance + 40% target variance
•	Output: Top-k most discriminative features
Training Strategy
1.	Pre-training: Models trained on source species data
2.	Fine-tuning: Selected models fine-tuned on combined source + target data
3.	Ensemble Weighting: Dynamic weighting based on validation performance
8. Output Files
Results Directory Structure
text
cross_species_results/
├── results_summary.json          # Summary metrics
├── detailed_results.json         # Comprehensive results
├── analysis_report.txt           # Text summary
├── machine_learning_results.xlsx # Excel results
├── Rat2Pig_model.joblib          # Rat→Pig trained model
├── Pig2Rat_model.joblib          # Pig→Rat trained model
├── Rat2Pig_cv_results.png        # Cross-validation plot
├── Rat2Pig_roc.png              # ROC curves
├── Rat2Pig_cm.png               # Confusion matrix
└── cross_species_analysis.log   # Execution log
Key Output Metrics
•	Accuracy: Overall classification accuracy
•	Precision: Weighted precision across classes
•	Recall: Weighted recall across classes
•	F1-Score: Harmonic mean of precision and recall
•	ROC-AUC: Area under ROC curve (multi-class)
•	Confusion Matrix: Class-wise performance
•	Cross-validation Scores: Training and test performance
9. Interpretation of Results
Performance Metrics
•	Accuracy > 0.8: Excellent cross-species transfer
•	Accuracy 0.6-0.8: Good predictive performance
•	Accuracy < 0.6: Limited cross-species applicability
Feature Selection
•	High-variance features: Conserved biological mechanisms
•	F-test significant features: Species-specific responses
•	Selected feature count: Optimal feature set size
Model Validation
•	Cross-validation consistency: Model reliability
•	Training-test gap: Overfitting assessment
•	Ensemble diversity: Complementary model strengths
10. Advanced Usage
Custom Data Integration
python
# For different species or time points
class_order = ['Control', '1h', '2h', '4h', '8h', '12h']  # Custom time points
Model Extension
python
# Add custom models to the ensemble
def _initialize_models(self):
    base_models = [
        # Existing models...
        ('CustomModel', YourCustomClassifier(parameters))
    ]
    return base_models
Feature Selection Modification
python
def adaptive_feature_selection(self, X_source, X_target, y_source, top_k=None):
    # Implement custom feature selection criteria
    custom_scores = your_custom_method(X_source, X_target, y_source)
    # Replace combined_scores calculation
11. Troubleshooting
Common Issues
1.	File Not Found Error
o	Verify file paths in AnalysisConfig
o	Check Excel file permissions
2.	Memory Errors
o	Reduce feature_selection_top_k
o	Decrease n_estimators
o	Use data sampling
3.	Poor Performance
o	Increase feature_selection_top_k
o	Adjust max_depth and n_estimators
o	Verify data quality and preprocessing
4.	Convergence Warnings
o	Increase max_iter for Logistic Regression
o	Adjust learning_rate for XGBoost
Debug Mode
Enable detailed logging by modifying the logging level in utils.py:
python
setup_logging(level=logging.DEBUG)
12. Citation
When using this framework in publications, please cite:
13. Support
For technical support or questions:
•	Check the log file: cross_species_analysis.log
•	Verify input data format requirements
•	Ensure all dependencies are correctly installed
•	Contact: 
•	Liangliang.wang@sxmu.edu.cn
•	Shanxi Medical University
14. License
This framework is available for academic use. For commercial applications, please contact the authors.
________________________________________
*This user manual corresponds to version 1.0 of the Cross-Species Transcriptomic Classification Framework. For updates and additional resources, please contact the author.
