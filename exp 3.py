import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class CowrieLogAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def load_cowrie_logs(self):
        """Load and parse Cowrie JSON logs"""
        data = []
        try:
            with open(self.file_path, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        data.append(log_entry)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        continue
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
            return []
        return data

    def extract_features(self, logs):
        """Extract relevant features from logs"""
        features = []
        for log in logs:
            feature_dict = {
                'eventid': log.get('eventid', 'unknown'),
                'src_ip': log.get('src_ip', 'unknown'),
                'timestamp': log.get('timestamp', ''),
                'input': log.get('input', ''),
                'success': 1 if log.get('eventid') == 'cowrie.login.success' else 0
            }
            features.append(feature_dict)
        return pd.DataFrame(features)

    def engineer_features(self, df):
        """Feature engineering without geolocation features"""
        # Timestamp features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Encode categorical variables
        if 'eventid' in df.columns:
            df['eventid'] = self.label_encoder.fit_transform(df['eventid'].astype(str))

        return df

    def label_attempts(self, df):
        """Label malicious attempts"""
        df['is_malicious'] = df['success'] == 0  # Example rule
        return df

    def prepare_data(self):
        """Prepare data for modeling"""
        logs = self.load_cowrie_logs()
        self.df = self.extract_features(logs)
        self.df = self.engineer_features(self.df)
        self.df = self.label_attempts(self.df)

        feature_columns = ['eventid', 'hour', 'day_of_week', 'is_weekend']
        self.X = self.df[feature_columns]
        self.y = self.df['is_malicious']

        # Scale features
        self.X = pd.DataFrame(self.scaler.fit_transform(self.X), columns=feature_columns)

    def train_and_evaluate_models(self):
        """Train and evaluate models"""
        if self.X is None or self.y is None or len(np.unique(self.y)) < 2:
            print("Error: Data not properly prepared or insufficient classes")
            return None

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42)
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }

        return results


analyzer = CowrieLogAnalyzer('cowrie2.json')
analyzer.prepare_data()
results = analyzer.train_and_evaluate_models()
print(results)