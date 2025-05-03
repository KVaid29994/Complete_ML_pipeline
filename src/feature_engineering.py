import pandas as pd
import numpy as np 
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import yaml


#create a logging function
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, 'feature_engineering.log'))
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

def load_params(params_path: str) -> dict:
    '''load parameters from a YAML file'''
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f"Parameters loaded from {params_path}.")
        return params
    except FileNotFoundError as e:
        logger.error(f"Parameters file not found: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise


def load_data(file_path:str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        # Check if the data is empty
        logger.info(f"Data loaded successfully from {file_path}")
        df.fillna(0, inplace=True)  # Fill NaN values with 0
        logger.info(f"NaN values filled with 0 in {file_path}")
        return df
    except pd.errors.ParserErrors as e:
        logger.error(f"Error parsing the CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading the CSV file: {e}")
        raise


def apply_tf_idf(train_data : pd.DataFrame, test_data : pd.DataFrame, max_features) -> tuple:
    """
    Apply TF-IDF vectorization to the specified column in the train and test data.
    """
    try:
        # Initialize the TF-IDF Vectorizer
        tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        X_train = train_data['text'].astype(str).values
        X_test = test_data['text'].astype(str).values
        y_train = train_data['target'].values
        y_test = test_data['target'].values

        X_train_bow = tfidf_vectorizer.fit_transform(X_train)
        X_test_bow = tfidf_vectorizer.transform(X_test)
        logger.info(f"TF-IDF vectorization applied to X_train and X_test data")

        train_df = pd.DataFrame(X_train_bow.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
        test_df['label'] = y_test

        logger.info(f"TF-IDF vectorization applied to train and test dataframes")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error applying Bag of words, {e}")
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save the DataFrame to a CSV file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug(f"Data saved successfully to {file_path}")

    except Exception as e:
        logger.error(f"Error saving the DataFrame to CSV: {e}")
        raise

def main():
    """
    Main function to load data, apply TF-IDF, and save the processed data.
    """
    try:
        params = load_params("params.yaml")
        # Load parameters from the YAML file
        max_features = params['feature_engineering']['max_features'] 
                # Load the data
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        # Apply TF-IDF vectorization
        train_df, test_df = apply_tf_idf(train_data, test_data, max_features)

        # Save the processed data
        save_data(train_df, os.path.join('./data','processed','train_tfidf.csv'))
        save_data(test_df, os.path.join('./data','processed','test_tfidf.csv'))

    except Exception as e:
        logger.error(f"Failed to complete feature engineering: {e}")
        raise

if __name__ == "__main__":
    main()