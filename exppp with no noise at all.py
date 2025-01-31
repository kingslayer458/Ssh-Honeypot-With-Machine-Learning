import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

class CowrieLogAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self.isolation_forest = None

    def load_and_prepare_data(self):
        """Main function to load and prepare data"""
        try:
            # Set random seed for reproducibility
            np.random.seed(42)
            
            # Load JSON data
            with open(self.file_path, 'r') as f:
                logs = [json.loads(line.strip()) for line in f]

            # Convert to DataFrame
            self.df = pd.DataFrame(logs)

            # Basic feature engineering
            self.engineer_features()

            # Label the data
            self.create_labels()

            # Prepare features for modeling
            self.prepare_features()

            print(f"Data prepared successfully. Shape: {self.X.shape}")
            return True

        except Exception as e:
            print(f"Error in data preparation: {e}")
            return False

    def engineer_features(self):
        """Engineer features from the log data"""
        # Event type features
        self.df['is_login_attempt'] = self.df['eventid'].str.contains('login', na=False).astype(int)
        self.df['is_failed_login'] = self.df['eventid'].str.contains('login.failed', na=False).astype(int)
        self.df['is_command'] = self.df['eventid'].str.contains('command', na=False).astype(int)
        self.df['is_failed_command'] = self.df['eventid'].str.contains('command.failed', na=False).astype(int)

        # Extract 'command' if part of the event data
        if 'input' in self.df.columns:
            # Filter for rows where there is a command and use the 'input' field
            self.df['command'] = self.df.apply(
                lambda row: row['input'] if 'command' in row['eventid'] else None,
                axis=1
            )
        else:
            print("Warning: no 'input' field found for extracting commands. Check data structure.")

        # Time-based features
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['is_night'] = ((self.df['hour'] >= 22) | (self.df['hour'] <= 5)).astype(int)

        # Session-based features
        session_counts = self.df['session'].value_counts()
        self.df['session_frequency'] = self.df['session'].map(session_counts)

        # IP-based features
        ip_counts = self.df['src_ip'].value_counts()
        self.df['ip_frequency'] = self.df['src_ip'].map(ip_counts)

    def create_labels(self):
        """Create labels for malicious activity"""
        # Initialize labels
        self.df['is_malicious'] = 0

        # Define conditions for malicious activity
        conditions = [
            (self.df['is_failed_login'] == 1) & (self.df['ip_frequency'] > 3),
            (self.df['is_failed_command'] == 1) & (self.df['session_frequency'] > 5),
            (self.df['ip_frequency'] > self.df['ip_frequency'].quantile(0.75)),
            (self.df['is_night'] == 1) & (self.df['is_failed_login'] == 1),
            (self.df['session_frequency'] > self.df['session_frequency'].quantile(0.8))
        ]

        # Apply conditions
        for condition in conditions:
            self.df.loc[condition, 'is_malicious'] = 1

    def prepare_features(self):
        """Prepare features for modeling"""
        # Select features for modeling
        feature_columns = [
            'is_login_attempt', 'is_failed_login', 'is_command',
            'is_failed_command', 'is_night', 'hour',
            'session_frequency', 'ip_frequency'
        ]

        # Create feature matrix
        self.X = self.df[feature_columns]

        # Scale features
        self.X = pd.DataFrame(self.scaler.fit_transform(self.X), columns=feature_columns)

        # Create target variable
        self.y = self.df['is_malicious']

    def train_and_evaluate(self):
        """Train and evaluate models"""
        if self.X is None or self.y is None:
            print("Data not prepared. Run load_and_prepare_data first.")
            return None

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Initialize models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=5,
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=42
            ),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }

        results = {}

        # Train and evaluate each model
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'y_test': y_test,
                'y_pred_proba': y_pred_proba
            }

        return results

    def plot_results(self, results):
        """Plot evaluation metrics"""
        if results is None:
            print("No results to plot.")
            return

        plt.figure(figsize=(15, 5))

        # Plot accuracies
        plt.subplot(1, 2, 1)
        accuracies = [results[model]['accuracy'] for model in results]
        plt.bar(results.keys(), accuracies)
        plt.title('Model Accuracies')
        plt.xticks(rotation=45)
        plt.ylabel('Accuracy')

        # Plot ROC curves
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

    def detect_anomalies(self):
        """Detect anomalies using Isolation Forest"""
        if self.X is None:
            print("Data not prepared. Run load_and_prepare_data first.")
            return None
        
        # Initialize Isolation Forest
        self.isolation_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        
        # Fit model
        anomalies = self.isolation_forest.fit_predict(self.X)
        
        # Add anomaly score -1 is outlier, 1 as normal
        self.df['anomaly'] = anomalies
        outliers = self.df[self.df['anomaly'] == -1]

        print(f"\nDetected {len(outliers)} anomalies in the dataset.")
        return outliers

    def generate_dashboard(self):
        """Generate insights and summary statistics dashboard"""
        print("\nDashboard Insights:")

        # Print initial Cowrie setup details
        print("Cowrie Setup Details:")
        print("Cowrie Honeypot initialized and configured for monitoring.")

        # SSH Connection details
        if not self.df.empty:
            conn_details = self.df.iloc[0]
            print(f"\nSSH Connection Details for First Entry:")
            print(f"Timestamp: {conn_details['timestamp']}")
            print(f"Source IP: {conn_details['src_ip']}")
            print(f"Session ID: {conn_details['session']}")

        # Summarize all successful logins
        successful_logins = self.df[self.df['eventid'] == 'cowrie.login.success']
        if not successful_logins.empty:
            print("\nAuthentication:")
            login_summary = successful_logins.groupby(['username', 'password']).size().reset_index(name='count')
            for _, row in login_summary.iterrows():
                print(f"Username: {row['username']}, Password: {row['password']} - Attempted {row['count']} times")
        else:
            print("\nNo successful logins recorded.")

        # Session duration analysis
        session_start = self.df['timestamp'].min()
        session_end = self.df['timestamp'].max()
        session_duration = (session_end - session_start).total_seconds()
        print("\nSession Duration:")
        print(f"Session started at {session_start} and ended at {session_end}, lasting {session_duration:.2f} seconds.")

        # Commands executed
        command_summary = self.df['command'].dropna().value_counts()
        print("\nCommands Executed:")
        if not command_summary.empty:
            for command, count in command_summary.items():
                print(f"Command: {command} - Executed {count} times.")
        else:
            print("No commands were executed.")

def main():
    # File path
    file_path = 'cowrie2.json'  # Replace with your log file path

    # Initialize analyzer
    analyzer = CowrieLogAnalyzer(file_path)

    # Load and prepare data
    if analyzer.load_and_prepare_data():
        # Print data statistics
        print("\nDataset Statistics:")
        print(f"Total samples: {len(analyzer.df)}")
        print(f"Malicious samples: {analyzer.y.sum()}")
        print(f"Benign samples: {len(analyzer.y) - analyzer.y.sum()}")

        # Train and evaluate models
        results = analyzer.train_and_evaluate()

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

            # Detect anomalies
            outliers = analyzer.detect_anomalies()
            print(f"Anomalies detected:\n{outliers}")

            # Generate dashboard
            analyzer.generate_dashboard()
        else:
            print("Error in model training and evaluation")
    else:
        print("Error in data preparation")

if __name__ == "__main__":
    main()