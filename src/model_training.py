import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import logging

## create logs directory
logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok=True)

logger = logging.getLogger('model_training')
logger.setLevel(logging.DEBUG)

counsoule = logging.StreamHandler()
counsoule.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(logs_dir, 'model_training.log'))
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
counsoule.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(counsoule)
logger.addHandler(file_handler) 


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
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


def train_model(X_train : np.ndarray, y_train: np.ndarray , params: dict) -> RandomForestClassifier:
    """
    Train a Random Forest model.
    """
    try:
        # Check if the input data is valid
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of samples in X and y must be the same")
        
        logger.debug("initializing RandomForestClassifier with parameters : %s", params)
        clf = RandomForestClassifier(n_estimators= params['n_estimators'],random_state=params['random_state'])
        logger.debug("Fitting the model")
        clf.fit(X_train, y_train)
        logger.debug("Model trained successfully")
        return clf
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

def save_model(model, file_path: str) -> None:
    """
    Save the trained model to a file.
    : param model: Trained model to save
    "param file_path: Path to save the model
    """
    try:
        #ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Save the model using pickle
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug(f"Model saved successfully to {file_path}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def main():
    try:
        params = {
                'n_estimators': 100,
                'random_state': 42
            }
            # load the training data
        train_data  = load_data('./data/processed/train_tfidf.csv')
            #spilt the data into features and target
        X_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values
        #set up a logger
        logger.debug("Training data loaded and split into features and target")
        logger.debug("X_train shape: %s, y_train shape: %s", X_train.shape, y_train.shape)

        #train the model 

        clf = train_model(X_train, y_train, params)
        model_save_path = './models/random_forest_model.pkl'
        save_model(clf, model_save_path)
        logger.debug("Model training and saving completed successfully")

    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")
        raise

if __name__ == "__main__":
    main()
    logger.debug("Starting the model training script")