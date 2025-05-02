import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score , precision_score, recall_score, roc_auc_score
import logging


# Set up logging configuration
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, 'model_evaluation.log'))
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

def load_model(file_path : str):
    """
    Load the model from the specified path.
    """
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug(f"Model loaded successfully from {file_path}")
        return model
    except FileExistsError:
        logger.error(f"Model file not found at {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def load_data(file_path : str):
    """
    Load the data from the specified path.
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Data loaded successfully from {file_path}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {e}")
        raise

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def evaluate_model(clf, X_test : np.ndarray, y_test : np.ndarray):
    """
    Evaluate the model using various metrics.
    """
    try:
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

        logger.debug(f"Model evaluation metrics: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, ROC AUC: {roc_auc}")
        
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc
        }
        logger.debug("Model evaluation metrics saved to dictionary")
        return metrics_dict
    
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise
def save_metrics(metrics_dict : dict, file_path : str):
    """
    Save the evaluation metrics to a JSON file.
    """
    #ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    #save the metrics to a json file

    try:
        with open(file_path, 'w') as file:
            json.dump(metrics_dict, file, indent=4)
        logger.debug(f"Metrics saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
        raise

def main():
    try:
        clf = load_model('./models/random_forest_model.pkl')
        test_data = load_data('./data/processed/test_tfidf.csv')

        X_test = test_data.iloc[:,:-1].values
        y_test = test_data.iloc[:,-1].values

        metrics = evaluate_model(clf, X_test, y_test)
        save_metrics(metrics, './metrics/model_evaluation.json')    
        logger.debug("Model evaluation completed successfully")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()
    logger.debug("Model evaluation script started")