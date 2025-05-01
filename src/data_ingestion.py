import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging

# Create logs directory
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# File handler
log_file_path = os.path.join(log_dir, "data_ingestion.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(data_url: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        data_url (str): URL or file path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    try:
        df = pd.read_csv(data_url)
        logger.debug(f"Data loaded from {data_url} with shape {df.shape}.")
        logger.info("Data loaded successfully.")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Error loading data: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by removing duplicates and handling missing values.

    Args:
        df (pd.DataFrame): Data to preprocess.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    try:
        initial_shape = df.shape
        df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True, axis=1)
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
        logger.debug(f"Columns renamed. Shape changed from {initial_shape} to {df.shape}.")
        return df
    except KeyError as e:
        logger.error("Missing columns in the DataFrame.")
        logger.error(f"KeyError: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred during preprocessing: {e}")
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """
    Save the train and test data to CSV files.

    Args:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.
        data_path (str): Directory path to save the data.
    """
    try:
        raw_data_path = os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug(f"Train and test data saved to {raw_data_path}.")
        logger.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}.")
        logger.info("Data saved successfully.")
    except Exception as e:
        logger.error(f"An error occurred while saving data: {e}")
        raise


def main():
    try:
        test_size = 0.2
        data_url = "https://raw.githubusercontent.com/KVaid29994/Complete_ML_pipeline/main/spam.csv"
        df = load_data(data_url)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path="./data")
        # You can add train_test_split or further steps here
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print (f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()
