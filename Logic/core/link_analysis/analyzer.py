import json
import random
from Logic.core.link_analysis.graph import LinkGraph
from Logic.core.indexer.indexes_enum import Indexes
from Logic.core.indexer.index_reader import Index_reader


def get_top_results(scores, max_result):
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:max_result]


def normalize_scores(scores):
    norm = sum(scores.values())
    for key in scores:
        scores[key] /= norm
    return scores


class LinkAnalyzer:
    def __init__(self, root_set):
        """
        Initialize the Link Analyzer attributes:

        Parameters
        ----------
        root_set: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names
        """
        self.document_index = Index_reader('../Indexes/', Indexes.DOCUMENTS).get_index()
        self.root_set = root_set
        self.graph = LinkGraph()
        self.hubs = {}
        self.authorities = {}
        self.initiate_params()

    def add_movie_node(self, movie):
        movie_node = ('movie', movie['id'])
        self.graph.add_node(movie_node)
        self.hubs[movie_node] = 1
        return movie_node

    def add_star_nodes_and_edges(self, movie_node, stars):
        for star in stars:
            star_node = ('star', star)
            self.graph.add_node(star_node)
            self.graph.add_edge(movie_node, star_node)
            self.graph.add_edge(star_node, movie_node)
            self.authorities[star_node] = 1

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        for movie in self.root_set:
            movie_node = self.add_movie_node(movie)
            if movie.get('stars'):
                self.add_star_nodes_and_edges(movie_node, movie['stars'])

    def add_node_if_absent(self, node, node_type):
        if node not in self.graph.graph:
            self.graph.add_node(node)
            if node_type == 'movie':
                self.hubs[node] = 1
            elif node_type == 'star':
                self.authorities[node] = 1

    def add_movie_and_stars(self, movie):
        movie_node = ('movie', movie['id'])
        self.add_node_if_absent(movie_node, 'movie')
        if movie.get('stars'):
            for star in movie['stars']:
                star_node = ('star', star)
                self.add_node_if_absent(star_node, 'star')
                self.graph.add_edge(movie_node, star_node)
                self.graph.add_edge(star_node, movie_node)

    def expand_graph(self, corpus):
        """
        expand hubs, authorities and graph using given corpus

        Parameters
        ----------
        corpus: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "stars": A list of movie star names

        Note
        -----
        To build the base set, we need to add the hubs and authorities that are inside the corpus
        and refer to the nodes in the root set to the graph and to the list of hubs and authorities.
        """
        for movie in corpus:
            self.add_movie_and_stars(movie)

    def update_authorities(self):
        new_star_authorities = {}
        for star_node in self.authorities:
            new_star_authorities[star_node] = sum(
                self.hubs[movie_node] for movie_node in self.graph.get_predecessors(star_node))
        return normalize_scores(new_star_authorities)

    def update_hubs(self):
        new_movie_hubs = {}
        for movie_node in self.hubs:
            new_movie_hubs[movie_node] = sum(
                self.authorities[star_node] for star_node in self.graph.get_successors(movie_node))
        return normalize_scores(new_movie_hubs)

    def extract_titles(self, movie_nodes):
        return [self.document_index[movie[0][1]]['title'] for movie in movie_nodes]

    def extract_names(self, star_nodes):
        return [star[0][1] for star in star_nodes]

    def hits(self, num_iteration=5, max_result=10):
        """
        Return the top movies and actors using the Hits algorithm

        Parameters
        ----------
        num_iteration: int
            Number of algorithm execution iterations
        max_result: int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            List of names of 10 actors with the most scores obtained by Hits algorithm in descending order
        list
            List of names of 10 movies with the most scores obtained by Hits algorithm in descending order
        """
        for _ in range(num_iteration):
            self.authorities = self.update_authorities()
            self.hubs = self.update_hubs()

        top_movies = get_top_results(self.hubs, max_result)
        top_stars = get_top_results(self.authorities, max_result)

        movie_titles = self.extract_titles(top_movies)
        star_names = self.extract_names(top_stars)

        return star_names, movie_titles


if __name__ == "__main__":
    # You can use this section to run and test the results of your link analyzer
    corpus = json.load(open('../IMDB_crawled.json'))
    root_set = random.sample(corpus, len(corpus) // 20)

    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=5)
    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')
