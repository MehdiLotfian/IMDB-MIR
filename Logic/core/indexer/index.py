import time
import os
import json
import copy
from indexes_enum import Indexes
from collections import defaultdict


class Index:
    def __init__(self, preprocessed_documents: list):
        """
        Create a class for indexing.
        """

        self.preprocessed_documents = preprocessed_documents

        self.index = {
            Indexes.DOCUMENTS.value: self.index_documents(),
            Indexes.STARS.value: self.index_stars(),
            Indexes.GENRES.value: self.index_genres(),
            Indexes.SUMMARIES.value: self.index_summaries(),
        }

    def index_documents(self):
        """
        Index the documents based on the document ID. In other words, create a dictionary
        where the key is the document ID and the value is the document.

        Returns
        ----------
        dict
            The index of the documents based on the document ID.
        """

        current_index = {doc['id']: doc for doc in self.preprocessed_documents}

        return current_index

    def index_stars(self):
        """
        Index the documents based on the stars.

        Returns
        ----------
        dict
            The index of the documents based on the stars. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        current_index = defaultdict(lambda: defaultdict(int))

        for document in self.preprocessed_documents:
            document_stars = document[Indexes.STARS.value]
            for star in document_stars:
                star_names = star.split()
                for name in star_names:
                    current_index[name][document['id']] += 1

        return current_index

    def index_genres(self):
        """
        Index the documents based on the genres.

        Returns
        ----------
        dict
            The index of the documents based on the genres. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        current_index = defaultdict(lambda: defaultdict(int))

        for document in self.preprocessed_documents:
            document_genres = document[Indexes.GENRES.value]
            for genre in document_genres:
                genre_parts = genre.split()
                for genre_part in genre_parts:
                    current_index[genre_part][document['id']] += 1

        return current_index

    def index_summaries(self):
        """
        Index the documents based on the summaries (not first_page_summary).

        Returns
        ----------
        dict
            The index of the documents based on the summaries. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        current_index = defaultdict(lambda: defaultdict(int))

        for document in self.preprocessed_documents:
            document_summaries = document[Indexes.SUMMARIES.value]
            for summary in document_summaries:
                summary_parts = summary.split()
                for summary_part in summary_parts:
                    current_index[summary_part][document['id']] += 1

        return current_index

    def get_posting_list(self, word: str, index_type: str):
        """
        get posting_list of a word

        Parameters
        ----------
        word: str
            word we want to check
        index_type: str
            type of index we want to check (documents, stars, genres, summaries)

        Return
        ----------
        list
            posting list of the word (you should return the list of document IDs that contain the word and ignore the tf)
        """

        try:
            return self.index[index_type][word]
        except:
            return []

    def add_document_to_index(self, document: dict):
        """
        Add a document to all the indexes

        Parameters
        ----------
        document : dict
            Document to add to all the indexes
        """

        index_types = [Indexes.DOCUMENTS, Indexes.STARS, Indexes.GENRES, Indexes.SUMMARIES]

        # Update DOCUMENTS index
        self.index[Indexes.DOCUMENTS.value][document['id']] = document

        # Update other indexes
        for index_type in index_types[1:]:
            field_values = document[index_type.value]
            for field_value in field_values:
                terms = field_value.split()
                for term in terms:
                    self.index[index_type.value][term][document['id']] += 1

    def remove_document_from_index(self, document_id: str):
        """
        Remove a document from all the indexes

        Parameters
        ----------
        document_id : str
            ID of the document to remove from all the indexes
        """

        document = self.index[Indexes.DOCUMENTS.value][document_id]
        self.index[Indexes.DOCUMENTS.value].pop(document_id)

        index_types = [Indexes.STARS, Indexes.GENRES, Indexes.SUMMARIES]

        for index_type in index_types:
            for field_value in document[index_type.value]:
                if isinstance(field_value, str):
                    terms = field_value.split()
                else:
                    terms = [field_value]
                for term in terms:
                    self._decrease_index(index_type, term, document_id)

    def _decrease_index(self, index_type, term, document_id):
        """
        Decrease the index entry for the given term and document ID.

        Parameters
        ----------
        index_type : Indexes
            The type of index to update.
        term : str
            The term to update.
        document_id : str
            The document ID.
        """
        self.index[index_type.value][term][document_id] -= 1
        if self.index[index_type.value][term][document_id] == 0:
            self.index[index_type.value][term].pop(document_id)

    def check_add_remove_is_correct(self):
        """
        Check if the add and remove is correct
        """

        dummy_document = {
            'id': '100',
            'stars': ['tim', 'henry'],
            'genres': ['drama', 'crime'],
            'summaries': ['good']
        }

        index_before_add = copy.deepcopy(self.index)
        self.add_document_to_index(dummy_document)
        index_after_add = copy.deepcopy(self.index)

        if index_after_add[Indexes.DOCUMENTS.value]['100'] != dummy_document:
            print('Add is incorrect, document')
            return

        if (set(index_after_add[Indexes.STARS.value]['tim']).difference(
                set(index_before_add[Indexes.STARS.value]['tim']))
                != {dummy_document['id']}):
            print('Add is incorrect, tim')
            return

        if (set(index_after_add[Indexes.STARS.value]['henry']).difference(
                set(index_before_add[Indexes.STARS.value]['henry']))
                != {dummy_document['id']}):
            print('Add is incorrect, henry')
            return
        if (set(index_after_add[Indexes.GENRES.value]['drama']).difference(
                set(index_before_add[Indexes.GENRES.value]['drama']))
                != {dummy_document['id']}):
            print('Add is incorrect, drama')
            return

        if (set(index_after_add[Indexes.GENRES.value]['crime']).difference(
                set(index_before_add[Indexes.GENRES.value]['crime']))
                != {dummy_document['id']}):
            print('Add is incorrect, crime')
            return

        if (set(index_after_add[Indexes.SUMMARIES.value]['good']).difference(
                set(index_before_add[Indexes.SUMMARIES.value]['good']))
                != {dummy_document['id']}):
            print('Add is incorrect, good')
            return

        print('Add is correct')

        self.remove_document_from_index('100')
        index_after_remove = copy.deepcopy(self.index)

        if index_after_remove == index_before_add:
            print('Remove is correct')
        else:
            print('Remove is incorrect')

    def store_index(self, path: str, index_name: str = None):
        """
        Stores the index in a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to store the file
        index_name: str
            name of index we want to store (documents, stars, genres, summaries)
        """

        if not os.path.exists(path):
            os.makedirs(path)

        if index_name is None:
            filename = 'index.json'
            data = self.index
        else:
            if index_name not in self.index:
                raise ValueError('Invalid index name')
            filename = f'{index_name}_index.json'
            data = self.index[index_name]

        with open(os.path.join(path, filename), 'w') as file:
            json.dump(data, file, indent=4)

    def load_index(self, path: str, index_name: str = None):
        """
        Loads the index from a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to load the file

        index_name : str, optional
            The name of the specific index to load. If None, loads the entire index.
        """

        try:
            if index_name is None:
                filename = 'index.json'
            else:
                filename = f'{index_name}_index.json'

            with open(os.path.join(path, filename), 'r') as file:
                loaded_index = json.load(file)

            if index_name is None:
                self.index = loaded_index
            else:
                self.index[index_name] = loaded_index

            return self.index[index_name] if index_name else self.index
        except Exception as e:
            print(f"Error loading index: {e}")
            return {}

    def check_if_index_loaded_correctly(self, index_type: str, loaded_index: dict):
        """
        Check if the index is loaded correctly

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        loaded_index : dict
            The loaded index

        Returns
        ----------
        bool
            True if index is loaded correctly, False otherwise
        """

        return self.index[index_type] == loaded_index if index_type else self.index == loaded_index

    def check_if_indexing_is_good(self, index_type: str, check_word: str = 'good'):
        """
        Checks if the indexing is good. Do not change this function. You can use this
        function to check if your indexing is correct.

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        check_word : str
            The word to check in the index

        Returns
        ----------
        bool
            True if indexing is good, False otherwise
        """

        # brute force to check check_word in the summaries
        start = time.time()
        docs = []
        for document in self.index[Indexes.DOCUMENTS.value].values():
            if index_type not in document or document[index_type] is None:
                continue

            for field in document[index_type]:
                if check_word in field:
                    docs.append(document['id'])
                    break

            # if we have found 3 documents with the word, we can break
            if len(docs) == 3:
                break

        end = time.time()
        brute_force_time = end - start

        # check by getting the posting list of the word
        start = time.time()
        posting_list = self.get_posting_list(check_word, index_type)

        end = time.time()
        implemented_time = end - start

        print('Brute force time: ', brute_force_time)
        print('Implemented time: ', implemented_time)

        if set(docs).issubset(set(posting_list)):
            print('Indexing is correct')

            if implemented_time < brute_force_time:
                print('Indexing is good')
                return True
            else:
                print('Indexing is bad')
                return False
        else:
            print('Indexing is wrong')
            return False


index = Index([])
for i in range(101, 49000):
    index.add_document_to_index(
        {'id': str(i), 'stars': ['actor', 'actor 2', 'actor 3'], 'genres': ['drama'], 'summaries': ['good summary']})
index.add_document_to_index(
    {'id': '50001', 'stars': ['tom', 'jack', 'max'], 'genres': ['action', 'crime'], 'summaries': ['bad summary']})
index.add_document_to_index(
    {'id': '50002', 'stars': ['tom', 'tom', 'max'], 'genres': ['action', 'crime'], 'summaries': ['bad summary']})
index.add_document_to_index(
    {'id': '50003', 'stars': ['jenny', 'alex', 'mohammad'], 'genres': ['biography'], 'summaries': ['a summary']})
index.add_document_to_index(
    {'id': '50004', 'stars': ['tom', 'alex', 'mohammad'], 'genres': ['biography'], 'summaries': ['a summary']})
index.add_document_to_index(
    {'id': '50005', 'stars': ['tom', 'alex', 'mohammad'], 'genres': ['biography'], 'summaries': ['a summary']})
index.add_document_to_index(
    {'id': '50006', 'stars': ['tom', 'tom', 'max'], 'genres': ['action', 'crime'], 'summaries': ['bad summary']})

index.store_index('indexes_files')
index.store_index('indexes_files', Indexes.DOCUMENTS.value)
index.store_index('indexes_files', Indexes.STARS.value)
index.store_index('indexes_files', Indexes.GENRES.value)
index.store_index('indexes_files', Indexes.SUMMARIES.value)
index.check_add_remove_is_correct()
print(index.check_if_index_loaded_correctly(None, index.load_index('indexes_files')))
print(
    index.check_if_index_loaded_correctly(Indexes.DOCUMENTS.value, index.load_index('indexes_files', Indexes.DOCUMENTS.value)))
print(index.check_if_index_loaded_correctly(Indexes.STARS.value, index.load_index('indexes_files', Indexes.STARS.value)))
print(index.check_if_index_loaded_correctly(Indexes.GENRES.value, index.load_index('indexes_files', Indexes.GENRES.value)))
print(
    index.check_if_index_loaded_correctly(Indexes.SUMMARIES.value, index.load_index('indexes_files', Indexes.SUMMARIES.value)))
index.check_if_indexing_is_good(Indexes.STARS.value, 'actor')
index.check_if_indexing_is_good(Indexes.STARS.value, 'tom')
index.check_if_indexing_is_good(Indexes.GENRES.value, 'drama')
index.check_if_indexing_is_good(Indexes.GENRES.value, 'biography')
index.check_if_indexing_is_good(Indexes.SUMMARIES.value, 'good')
index.check_if_indexing_is_good(Indexes.SUMMARIES.value, 'bad')
