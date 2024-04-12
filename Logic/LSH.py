import itertools
import random
from collections import defaultdict
import numpy as np


class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes
        self.shingles = [self.shingle_document(document) for document in self.documents]
        self.hash_funcs = [lambda x: hash((10 ** 9 + 9, i, x)) for i in range(num_hashes)]

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        words = document.split()
        return set(" ".join(words[i: i + k]) for i in range(len(words) - k + 1))

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        shingle_set = set(shingle for doc_shingles in self.shingles for shingle in doc_shingles)

        matrix = np.zeros((len(shingle_set), len(self.shingles)))

        unique_shingles = list(shingle_set)

        for i, shingle in enumerate(unique_shingles):
            for j, doc_shingles in enumerate(self.shingles):
                if shingle in doc_shingles:
                    matrix[i, j] = 1

        return matrix

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        hashes = []
        for doc in self.documents:
            hash_values = [min(map(hash_func, self.shingle_document(doc))) for hash_func in self.hash_funcs]
            hashes.append(np.array(hash_values))
        return hashes

    def lsh_buckets(self, signatures: np.ndarray, bands=10, rows_per_band=10):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signatures : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        buckets = defaultdict(list)
        for i, sig in enumerate(signatures):
            for band in range(bands):
                band_hash = hash(tuple(sig[band * rows_per_band: (band + 1) * rows_per_band]))
                buckets[band_hash].append(i)
        return buckets

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        bands = self.num_hashes // 5
        signatures = self.min_hash_signature()
        return self.lsh_buckets(signatures, bands, 5)

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        intersection_size = len(first_set & second_set)
        union_size = len(first_set | second_set)
        jaccard_score = intersection_size / union_size if union_size != 0 else 0
        return jaccard_score

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)


if __name__ == "__main__":
    import json

    with open("LSHFakeData.json", "r") as f:
        documents = json.load(f)
    document_summaries = [" # ".join(doc["summaries"]) for doc in documents]
    minhash_lsh = MinHashLSH(document_summaries, 500)
    minhash_lsh.jaccard_similarity_test(minhash_lsh.perform_lsh(), document_summaries)
