from collections import defaultdict
import heapq

class SpellCorrection:
    def __init__(self, all_documents):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str
            The input documents.
        """
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(all_documents)

    def shingle_word(self, word, k=2):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        shingles = set()

        word = f'${word}$'
        shingles = {word[i: i + k] for i in range(len(word) - k + 1)} if len(word) >= k else {word}

        return shingles
    
    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """

        intersection_size = len(first_set.intersection(second_set))
        union_size = len(first_set.union(second_set))

        jacard_score = 0.0

        if union_size != 0:
            jacard_score = intersection_size / union_size

        return jacard_score


    def shingling_and_counting(self, all_documents):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_shingled_words = {}
        word_counter = defaultdict(lambda: [0] * len(all_documents))

        for i, doc in enumerate(all_documents):
            words = doc.split()
            for word in words:
                all_shingled_words.setdefault(word, self.shingle_word(word))
                word_counter[word][i] += 1

        return all_shingled_words, word_counter
    
    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : stf
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        top5_candidates = list()

        word_shingles = self.shingle_word(word)
        top5 = [(0, '')] * 5
        for correct, shingles in self.all_shingled_words.items():
            if abs(len(correct) - len(word)) > 2:
                continue
            score = self.jaccard_score(word_shingles, shingles)
            heapq.heappushpop(top5, (score, correct))
        top5_candidates = [candidate[1] for candidate in top5 if candidate[1]]

        return top5_candidates
    
    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : stf
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        final_result = ""

        words = query.split()
        final_words = []
        for word in words:
            candidates = self.find_nearest_words(word)
            tf_scores = [sum(self.word_counter[x]) for x in candidates]
            max_tf_score = max(tf_scores, default=1)
            tf_scores_normalized = [score / max_tf_score for score in tf_scores]
            scores = [
                (
                    tf_scores_normalized[i] * self.jaccard_score(self.all_shingled_words[candidates[i]],
                                                                 self.shingle_word(word)),
                    candidates[i]
                ) for i in range(len(candidates))
            ]
            best_candidate = max(scores, key=lambda x: x[0])[1]
            final_words.append(best_candidate)

        final_result = " ".join(final_words)
        return final_result