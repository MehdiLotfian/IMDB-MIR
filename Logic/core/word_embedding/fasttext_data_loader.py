import nltk
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize

nltk.download('stopwords')


def preprocess_text(text, minimum_length=1, stopword_removal=True, stopwords_domain=None, lower_case=True,
                    punctuation_removal=True):
    """
    preprocess text by removing stopwords, punctuations, and converting to lowercase, and also filter based on a min length
    for stopwords use nltk.corpus.stopwords.words('english')
    for punctuations use string.punctuation

    Parameters
    ----------
    text: str
        text to be preprocessed
    minimum_length: int
        minimum length of the token
    stopword_removal: bool
        whether to remove stopwords
    stopwords_domain: list
        list of stopwords to be removed base on domain
    lower_case: bool
        whether to convert to lowercase
    punctuation_removal: bool
        whether to remove punctuations
    """
    if stopwords_domain is None:
        stopwords_domain = set(stopwords.words('english'))
    text = word_tokenize(text)

    if lower_case:
        text = [word.lower() for word in text]
    if punctuation_removal:
        text = [word for word in text if word not in string.punctuation]
    if stopword_removal:
        text = [word for word in text if (word.lower() not in stopwords_domain)]
    if minimum_length > 1:
        text = [word for word in text if len(word) >= minimum_length]

    text = " ".join(text)
    return text


def concatenate_row(row):
    if len(row['reviews']) == 0:
        return None
    synopses_text = ' '.join(row['synposis'])
    summaries_text = ' '.join(row['summaries'])
    reviews_text = ' '.join(row['reviews'][0])
    title_text = row['title']
    return ' '.join([synopses_text, summaries_text, reviews_text, title_text])


class FastTextDataLoader:
    """
    This class is designed to load and pre-process data for training a FastText model.

    It takes the file path to a data source containing movie information (synopses, summaries, reviews, titles, genres) as input.
    The class provides methods to read the data into a pandas DataFrame, pre-process the text data, and create training data (features and labels)
    """

    def __init__(self, file_path):
        """
        Initializes the FastTextDataLoader class with the file path to the data source.

        Parameters
        ----------
        file_path: str
            The path to the file containing movie information.
        """
        self.file_path = file_path

    def read_data_to_df(self):
        """
        Reads data from the specified file path and creates a pandas DataFrame containing movie information.

        You can use an IndexReader class to access the data based on document IDs.
        It extracts synopses, summaries, reviews, titles, and genres for each movie.
        The extracted data is then stored in a pandas DataFrame with appropriate column names.

        Returns
        ----------
            pd.DataFrame: A pandas DataFrame containing movie information (synopses, summaries, reviews, titles, genres)
        """
        df = pd.read_json(self.file_path)
        columns_to_keep = ["synposis", "summaries", "reviews", "title", "genres"]
        df = df[columns_to_keep]
        return df

    def create_train_data(self):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).

        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
        df = self.read_data_to_df()
        columns_to_keep = ["synposis", "summaries", "reviews", "title", "genres"]
        df = df.dropna(subset=columns_to_keep)

        df['text'] = df.apply(concatenate_row, axis=1)
        df = df.dropna(subset=['text'])

        X = df['text'].apply(lambda x: preprocess_text(x)).values

        df['genres_string'] = df.apply(lambda row: ' '.join(row['genres']), axis=1)
        le = LabelEncoder()
        y = le.fit_transform(df['genres_string'])

        return X, y


# test the class
#fasttext_data_loader = FastTextDataLoader('..\\IMDB_crawled.json')
#X, y = fasttext_data_loader.create_train_data()
#print(X[0])
#print(y[0])
