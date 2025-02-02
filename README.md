# ssh-honeypot-with-machine-learning
# further explanation will be provided soon!!!!
# Cowrie Log Analyzer

## Introduction
Cowrie Log Analyzer is a powerful tool designed to analyze SSH honeypot logs collected from the [Cowrie Honeypot](https://github.com/cowrie/cowrie). This script extracts features from the logs, detects malicious activity, and evaluates various machine learning models to classify attacks.

## Features
- Parses Cowrie log files in JSON format.
- Extracts meaningful features for machine learning.
- Implements multiple classification algorithms:
  - Random Forest
  - Support Vector Machine (SVM)
  - Naive Bayes
  - Decision Tree
  - XGBoost
  - LightGBM
- Detects anomalies using Isolation Forest.
- Generates an interactive dashboard with insights.
- Visualizes classification performance and ROC curves.

## Installation

### Prerequisites
Ensure you have Python installed (Python 3.7 or later recommended). Install required dependencies:

```bash
pip install numpy pandas scikit-learn xgboost lightgbm matplotlib
```

## Usage

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/cowrie-log-analyzer.git
cd cowrie-log-analyzer
```

### 2. Prepare the Log File
Ensure you have a valid Cowrie JSON log file. By default, the script expects `cowrie.json`, but you can change this in the script.

### 3. Run the Analyzer
```bash
python expppp.py
```

## How It Works

### 1. Data Preprocessing
- Reads Cowrie log entries and converts them into structured data.
- Extracts relevant features (e.g., login attempts, failed logins, commands executed).
- Adds session and IP frequency counts for anomaly detection.

### 2. Labeling and Feature Engineering
- Labels log entries as malicious or benign based on patterns.
- Introduces controlled noise to simulate real-world data variations.

### 3. Model Training and Evaluation
- Splits data into training and test sets.
- Trains multiple machine learning models and evaluates their performance.
- Displays accuracy, confusion matrix, and classification reports.

### 4. Anomaly Detection
- Uses Isolation Forest to detect unusual activities.
- Flags potential malicious IPs and sessions.

### 5. Dashboard and Visualization
- Displays insights such as top login attempts, frequent attackers, and common commands executed.
- Plots model performance using ROC curves and accuracy comparison charts.

## Example Output
```plaintext
Dataset Statistics:
Total samples: 5000
Malicious samples: 1200
Benign samples: 3800

Results for Random Forest:
Accuracy: 0.91
Classification Report:
               precision    recall  f1-score   support

           0       0.92      0.94      0.93      1520
           1       0.88      0.85      0.87       480

Confusion Matrix:
[[1428   92]
 [  72  408]]
```

## Contribution
Feel free to contribute by submitting pull requests or reporting issues.

## License
This project is licensed under the MIT License.

## References
- [Cowrie Honeypot](https://github.com/cowrie/cowrie)
- [Scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/)
- [LightGBM](https://lightgbm.readthedocs.io/en/latest/)

