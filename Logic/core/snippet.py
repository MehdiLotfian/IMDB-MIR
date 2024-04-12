from Logic.core.preprocess import Preprocessor

class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side

    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """

        preprocessor = Preprocessor([])
        return preprocessor.remove_stopwords(query)

    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        query = self.remove_stop_words_from_query(query)
        final_snippet = ""
        query_words = query.split()
        doc_words = doc.split()

        not_exist_words = [word for word in query_words if word not in doc_words]

        arr = [query_words.index(word) if word in query_words else -1 for word in doc_words]
        index_arr = [i for i, n in enumerate(arr) if n != -1]

        while index_arr:
            keywords, end, start = {}, -1, float('inf')

            for n in index_arr:
                s = max(0, n - self.number_of_words_on_each_side)
                e =  min(len(arr),n + 1 + self.number_of_words_on_each_side)

                sub = set(arr[s:e])
                sub.discard(-1)
                if len(sub) > len(keywords) or (len(sub) == len(keywords) and e - s > end - start):
                    keywords, end, start = sub, e, s

            index_arr = [x for x in index_arr if not (arr[x] in keywords or start <= x < end)]
            arr = [-1 if x in keywords else x for x in arr]

            snippet = ['***' + word + '***' if word in query_words else word for word in doc_words[start:end]]
            final_snippet += ' '.join(snippet) + '...'

        final_snippet = final_snippet[3:] if final_snippet[:3] == '...' else final_snippet

        return final_snippet, not_exist_words
