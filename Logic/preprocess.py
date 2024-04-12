import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

nltk.download('wordnet')

class Preprocessor:

    def __init__(self, documents: list):
        """
        Initialize the class.

        Parameters
        ----------
        documents : list
            The list of documents to be preprocessed, path to stop words, or other parameters.
        """
        self.documents = documents
        self.stopwords = list(map(lambda x: x.replace('\n', ''), open('stopwords.txt', 'r').readlines()))

    def preprocess(self):
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        List[str]
            The preprocessed documents.
        """
        preprocessed_documents = []
        for document in self.documents:
            preprocessed_document = document.lower()
            preprocessed_document = self.remove_links(preprocessed_document)
            preprocessed_document = self.remove_punctuations(preprocessed_document)
            preprocessed_document = self.remove_stopwords(preprocessed_document)
            preprocessed_document = self.normalize(preprocessed_document)

            preprocessed_documents.append(preprocessed_document)
        return preprocessed_documents

    def normalize(self, text: str):
        """
        Normalize the text by converting it to a lower case, stemming, lemmatization, etc.

        Parameters
        ----------
        text : str
            The text to be normalized.

        Returns
        ----------
        str
            The normalized text.
        """
        text = text.lower()

        lem = WordNetLemmatizer()
        words = self.tokenize(text)
        words = [lem.lemmatize(word) for word in words]
        text = ' '.join(words)
        return text

    def remove_links(self, text: str):
        """
        Remove links from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with links removed.
        """
        patterns = [r'\S*http\S*', r'\S*www\S*', r'\S+\.ir\S*', r'\S+\.com\S*', r'\S+\.org\S*', r'\S*@\S*']
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        return text

    def remove_punctuations(self, text: str):
        """
        Remove punctuations from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with punctuations removed.
        """
        words = self.tokenize(text)
        words = [word for word in words if word not in string.punctuation]
        text = ' '.join(words)
        return text

    def tokenize(self, text: str):
        """
        Tokenize the words in the text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        ----------
        list
            The list of words.
        """
        return word_tokenize(text)

    def remove_stopwords(self, text: str):
        """
        Remove stopwords from the text.

        Parameters
        ----------
        text : str
            The text to remove stopwords from.

        Returns
        ----------
        str
            The text with stopwords removed.
        """
        words = self.tokenize(text)
        words = [word for word in words if word not in self.stopwords]
        text = ' '.join(words)
        return text

