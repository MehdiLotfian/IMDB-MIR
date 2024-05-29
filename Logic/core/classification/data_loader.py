import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ..word_embedding.fasttext_model import FastText
from ..word_embedding.fasttext_model import preprocess_text


class ReviewLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.fasttext_model = None
        self.review_tokens = []
        self.sentiments = []
        self.embeddings = []

    def load_data(self):
        """
        Load the data from the csv file and preprocess the text. Then save the normalized tokens and the sentiment labels.
        Also, load the fasttext model.
        """
        data = pd.read_csv(self.file_path)

        # Assuming the dataframe has 'review' and 'sentiment' columns
        self.review_tokens = data['review'].astype(str).tolist()
        sentiments = data['sentiment'].astype(str).tolist()

        self.sentiments = sentiments

        # Load FastText model
        fasttext = FastText()
        fasttext.prepare(None, 'load', path='C:/Users/mehdi/IMDB-MIR/Logic/core/word_embedding/FastText_model.bin')
        self.fasttext_model = fasttext.model

    def get_embeddings(self):
        """
        Get the embeddings for the reviews using the fasttext model.
        """
        if not self.fasttext_model:
            raise ValueError("FastText model is not loaded. Call load_data() first.")

        self.embeddings = []
        for tokens in tqdm(self.review_tokens):
            sentence_vector = self.fasttext_model.get_sentence_vector(tokens)
            self.embeddings.append(sentence_vector)

        self.embeddings = np.array(self.embeddings)

    def split_data(self, test_data_ratio=0.2):
        """
        Split the data into training and testing data.

        Parameters
        ----------
        test_data_ratio: float
            The ratio of the test data
        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            Return the training and testing data for the embeddings and the sentiments.
            in the order of x_train, x_test, y_train, y_test
        """
        if len(self.embeddings) == 0:
            raise ValueError("Embeddings are not generated. Call get_embeddings() first.")

        le = LabelEncoder()
        y_encoded = le.fit_transform(self.sentiments)

        x_train, x_test, y_train, y_test = train_test_split(
            self.embeddings, y_encoded, test_size=test_data_ratio, random_state=42
        )

        return x_train, x_test, y_train, y_test
