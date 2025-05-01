import os
import logging
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import string

nltk.download('stopwords')
nltk.download('punkt')

#ensure the logs directory exists
log_dir = 'logs'
os.makedirs('logs', exist_ok=True)

#setting up logging configuration
logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, "data_preprocessing.log"))
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """
    Function to transform text by removing punctuation, converting to lowercase,
    removing stopwords, and stemming the words.
    """
    # Initialize the stemmer
    ps = PorterStemmer()
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords and stem the words
    stop_words = set(stopwords.words('english'))
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def preprocess_df(df,text_column ="text", target_column = "target"):
    """
    Function to preprocess the dataframe by transforming the text and encoding the labels.
    """
    try:
        logger.debug("Starting preprocessing of the dataframe.")
        #encode the target labels
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug("Encoded target labels.")

        #remove duplicates rows
        df.drop_duplicates(keep = 'first',inplace=True)
        logger.debug("Removed duplicate rows.")

        #apply the text transformation function to the text column
        df[text_column] = df[text_column].apply(transform_text)
        logger.debug("Transformed text data.")
        return df
    except KeyError as e:
        logger.error(f"Error in preprocessing dataframe: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in preprocessing dataframe: {e}")
        raise

def main(text_column="text", target_column="target"):
    """
    Main function to run the preprocessing on the dataset.
    """
    try:
        # Load the dataset
        train_data = pd.read_csv("./data/raw/train.csv")
        logger.debug("Loaded train data.")
        test_data = pd.read_csv("./data/raw/test.csv")
        logger.debug("Loaded test data.")

        # apply the preprocessing function to the train and test dataframes
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        logger.debug("Preprocessed train data.")
        test_processed_data = preprocess_df(test_data, text_column, target_column) 
        logger.debug("Preprocessed test data.")

        # store data inside the data/processed directory
        data_path = os.path.join("./data","interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        logger.debug("Saved preprocessed train data.")
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        logger.debug("Saved preprocessed test data.")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
    logger.info("Preprocessing completed successfully.")