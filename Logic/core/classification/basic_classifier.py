import numpy as np
from tqdm import tqdm

from Logic.core.word_embedding.fasttext_model import FastText


class BasicClassifier:
    def __init__(self):
        pass

    def fit(self, x, y):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def prediction_report(self, x, y):
        raise NotImplementedError()

    def get_percent_of_positive_reviews(self, sentences):
        """
        Get the percentage of positive reviews in the given sentences
        Parameters
        ----------
        sentences: list
            The list of sentences to get the percentage of positive reviews
        Returns
        -------
        float
            The percentage of positive reviews
        """
        predictions = self.predict(sentences)
        positive_reviews = sum(predictions)
        percent_positive = positive_reviews / len(predictions) * 100
        return percent_positive

