import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree
import xgboost as xgb  # Import XGBoost
import lightgbm as lgb  # Import LightGBM
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
                'src_port': log.get('src_port', 0),
                'dst_port': log.get('dst_port', 0),
                'protocol': log.get('protocol', 'unknown'),
                'username': log.get('username', 'unknown'),
                'password': log.get('password', 'unknown'),
                'session': log.get('session', 'unknown'),
                'sensor': log.get('sensor', 'unknown'),
                'timestamp': log.get('timestamp', ''),
                'duration': log.get('duration', 0),
                'input': log.get('input', ''),
                'success': 1 if log.get('eventid') == 'cowrie.login.success' else 0
            }
            features.append(feature_dict)
        return pd.DataFrame(features)

    def engineer_features(self, df):
        """Perform feature engineering"""
        # Create event type features before encoding
        df['is_login_failed'] = df['eventid'].str.contains('login.failed', na=False).astype(int)
        df['is_login_success'] = df['eventid'].str.contains('login.success', na=False).astype(int)
        df['is_command_failed'] = df['eventid'].str.contains('command.failed', na=False).astype(int)
        df['is_command_input'] = df['eventid'].str.contains('command.input', na=False).astype(int)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)

        # Session-based features
        df['session_count'] = df.groupby('session')['session'].transform('count')
        df['commands_per_session'] = df['is_command_input'].groupby(df['session']).transform('sum')

        # IP-based features
        df['ip_attempts'] = df.groupby('src_ip')['src_ip'].transform('count')
        
        # Login attempt features
        df['login_attempts'] = df['is_login_failed'].groupby(df['src_ip']).transform('sum')
        
        # Convert categorical variables
        categorical_columns = ['protocol', 'sensor']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = self.label_encoder.fit_transform(df[col].astype(str))

        return df

    def label_attempts(self, df):
        """Label malicious attempts"""
        df['is_malicious'] = 0

        # Define malicious patterns
        conditions = [
            (df['is_login_failed'] == 1),
            (df['is_command_failed'] == 1),
            (df['ip_attempts'] > df['ip_attempts'].quantile(0.95)),
            (df['session_count'] > df['session_count'].quantile(0.95)),
            (df['commands_per_session'] > 10),
            (df['login_attempts'] > 3)
        ]

        # Set is_malicious to 1 if any condition is met
        df['is_malicious'] = np.where(np.any(conditions, axis=0), 1, 0)

        return df

    def prepare_data(self):
        """Prepare data for modeling"""
        logs = self.load_cowrie_logs()
        self.df = self.extract_features(logs)
        self.df = self.engineer_features(self.df)
        self.df = self.label_attempts(self.df)

        feature_columns = [
            'protocol', 'hour', 'day_of_week', 'is_weekend', 
            'is_night', 'ip_attempts', 'session_count', 'commands_per_session',
            'is_command_input', 'is_command_failed', 'is_login_failed', 
            'is_login_success', 'login_attempts'
        ]

        self.X = self.df[feature_columns]
        self.y = self.df['is_malicious']

        # Scale features
        self.X = pd.DataFrame(self.scaler.fit_transform(self.X), columns=feature_columns)

    def train_and_evaluate_models(self):
        """Train and evaluate multiple models"""
        if len(np.unique(self.y)) < 2:
            print("Warning: Only one class present in the dataset. Cannot train models.")
            return None

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=42),  # Added Decision Tree
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),  # Added XGBoost
            'LightGBM': lgb.LGBMClassifier(random_state=42)  # Added LightGBM
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'y_test': y_test,
                'y_pred_proba': y_pred_proba
            }

        return results

    def plot_results(self, results):
        """Plot evaluation metrics and ROC curves"""
        if results is None:
            print("No results to plot.")
            return

        plt.figure(figsize=(15, 5))

        # Plot 1: Model Accuracies
        plt.subplot(1, 2, 1)
        accuracies = [results[model]['accuracy'] for model in results]
        plt.bar(results.keys(), accuracies)
        plt.title('Model Accuracies')
        plt.xticks(rotation=45)
        plt.ylabel('Accuracy')

        # Plot 2: ROC Curves
        plt.subplot(1, 2, 2)
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.tight_layout()
        plt.show()

def main():
    # Initialize analyzer
    analyzer = CowrieLogAnalyzer('cowrie2.json')  # Replace with your log file path
    
    # Prepare data
    analyzer.prepare_data()
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(analyzer.df)}")
    print(f"Malicious samples: {analyzer.y.sum()}")
    print(f"Benign samples: {len(analyzer.y) - analyzer.y.sum()}")
    
    # Train and evaluate models
    results = analyzer.train_and_evaluate_models()
    
    if results:
        # Print results
        for model_name, metrics in results.items():
            print(f"\nResults for {model_name}:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print("\nClassification Report:")
            print(metrics['classification_report'])
            print("\nConfusion Matrix:")
            print(metrics['confusion_matrix'])
        
        # Plot results
        analyzer.plot_results(results)

if __name__ == "__main__":
    main()
