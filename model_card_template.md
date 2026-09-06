# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Developed by: Alicia Higgs
Model Date: September 6, 2026
Model Type: Random Forest Classifier
Model Version: Version 1.0 of the Census Income Prediction Model
References: Trained using scikit-learn on the UCI Adult Census Income dataset.

## Intended Use
Intended Use: This model should be used to predict whether a person earns more than $50,000 annually using standard demographic and employment features.
Intended Users: Students, machine learning engineers, and developers lookinh at examples of scalable machine learning pipelines and model deployment.
Out of Scope Use Cases: This is strictly an educational and portfolio project. This model should not be used for automated legal, financial, or credit-lending decisions.

## Training Data
Dataset: Training data is built from an 80% split of the UCI Adult Census Income dataset (`census.csv`).
Features: Combines categorical features (such as `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, and `native-country`) and numerical features.
Target Variable: `salary` (binary classification: `<=50K` or `>50K`).

## Evaluation Data
Dataset: Evaluation data consists of the remaining 20% held-out test split from the `census.csv` dataset, processed using the same OneHotEncoder fitted on the training set.

## Metrics
Metrics Used: Precision, Recall, and F1-Score. Because the income distribution is imbalanced, accuracy is not the best yardstick. Instead, I evaluated the model using
  Precision: 0.7353
  Recall: 0.6378
  F1-Score: 0.6831
Slice Performance: Detailed performance metrics across specific categorical subgroups (such as workclass and education) are logged and archived in `slice_output.txt`.

## Ethical Considerations
Bias and Fairness: Census datasets reflect historical socioeconomic biases. As a result, predictions from this model may reflect demographic disparities. This model should not be deployed in real-world sensitive applications without a thorough fairness and bias audit.
Data Privacy: The dataset consists of publicly available demographic records, but ethical deployment requires ensuring user privacy and complying with data governance standards.


## Caveats and Recommendations
Caveats: Because the model is trained on historical data, it may not translate well to modern economic landscapes or populations with different demographic makeups.
Recommendations: Users should review the slice performance metrics in `slice_output.txt` to understand how the model behaves across different demographic sub-segments before considering any extended application.