from requests import get
from bs4 import BeautifulSoup
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
import json


class IMDbCrawler:
    """
    put your own user agent in the headers
    """
    headers = {
        'User-Agent': "Random"
    }
    top_250_URL = 'https://www.imdb.com/chart/top/'

    def __init__(self, crawling_threshold=1000):
        """
        Initialize the crawler

        Parameters
        ----------
        crawling_threshold: int
            The number of pages to crawl
        """
        self.crawling_threshold = crawling_threshold
        self.not_crawled = deque()
        self.crawled = []
        self.added_ids = set()
        self.add_list_lock = Lock()
        self.add_queue_lock = Lock()
        self.crawler_counter_lock = Lock()
        self.crawled_counter = 0

    def get_id_from_URL(self, URL):
        """
        Get the id from the URL of the site. The id is what comes exactly after title.
        for example the id for the movie https://www.imdb.com/title/tt0111161/?ref_=chttp_t_1 is tt0111161.

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        str
            The id of the site
        """
        return URL.split('/')[2]

    def write_to_file_as_json(self):
        """
        Save the crawled files into json
        """
        with open('IMDB_crawled.json', 'w') as f:
            json.dump(list(self.crawled), f)

        with open('IMDB_not_crawled.json', 'w') as f:
            json.dump(list(self.not_crawled), f)

        with open('IMDB_added_ids.json', 'w') as f:
            json.dump(list(self.added_ids), f)

    def read_from_file_as_json(self):
        """
        Read the crawled files from json
        """
        with open('IMDB_crawled.json', 'r') as f:
            self.crawled = set(json.load(f))

        with open('IMDB_not_crawled.json', 'r') as f:
            self.not_crawled = deque(json.load(f))

        with open('IMDB_added_ids.json', 'r') as f:
            self.added_ids = set(json.load(f))

    def crawl(self, URL):
        """
        Make a get request to the URL and return the response

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        requests.models.Response
            The response of the get request
        """
        response = get(URL, headers=self.headers)
        if response.status_code == 200:
            return response
        else:
            return None

    def extract_top_250(self):
        """
        Extract the top 250 movies from the top 250 page and use them as seed for the crawler to start crawling.
        """
        response = self.crawl(self.top_250_URL)
        if response:
            soup = BeautifulSoup(response.text, 'html.parser')
            top_250_movie_links = soup.find_all(class_="ipc-title-link-wrapper")
            for movie_link in top_250_movie_links:
                movie_id = self.get_id_from_URL(movie_link['href'])
                with self.add_queue_lock:
                    self.not_crawled.append('https://www.imdb.com/title/' + movie_id + '/')
                    self.added_ids.add(movie_id)

    def get_imdb_instance(self):
        return {
            'id': None,  # str
            'title': None,  # str
            'first_page_summary': None,  # str
            'release_year': None,  # str
            'mpaa': None,  # str
            'budget': None,  # str
            'gross_worldwide': None,  # str
            'rating': None,  # str
            'directors': None,  # List[str]
            'writers': None,  # List[str]
            'stars': None,  # List[str]
            'related_links': None,  # List[str]
            'genres': None,  # List[str]
            'languages': None,  # List[str]
            'countries_of_origin': None,  # List[str]
            'summaries': None,  # List[str]
            'synopsis': None,  # List[str]
            'reviews': None,  # List[List[str]]
        }

    def start_crawling(self):
        """
        Start crawling the movies until the crawling threshold is reached.
        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """
        self.extract_top_250()
        futures = []
        crawled_counter = 0

        with ThreadPoolExecutor(max_workers=20) as executor:
            while crawled_counter < self.crawling_threshold:
                URL = self.not_crawled.popleft()
                futures.append(executor.submit(self.crawl_page_info, URL))
                if len(self.not_crawled) == 0:
                    wait(futures)
                    futures = []
                with self.crawler_counter_lock:
                    crawled_counter += 1
    def crawl_page_info(self, URL):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.

        Parameters
        ----------
        URL: str
            The URL of the site
        """
        # print("new iteration")
        movie = self.get_imdb_instance()
        response = self.crawl(URL)
        self.extract_movie_info(response, movie, URL)
        with self.add_list_lock:
            self.crawled.append(movie)
            self.crawled_counter += 1
        print(f'crawled {self.crawled_counter} of total {self.crawling_threshold}')
        for movie_link in movie['related_links']:
            self.add_movie_to_not_crawled(movie_link)

    def add_movie_to_not_crawled(self, movie_link):
        movie_id = movie_link.split('/')[4]
        if movie_id not in self.added_ids:
            self.not_crawled.append('https://www.imdb.com/title/' + movie_id + '/')
            self.added_ids.add(movie_id)

    def extract_movie_info(self, res, movie, URL):
        """
        Extract the information of the movie from the response and save it in the movie instance.

        Parameters
        ----------
        res: requests.models.Response
            The response of the get request
        movie: dict
            The instance of the movie
        URL: str
            The URL of the site
        """
        if res:
            movie_soup = BeautifulSoup(res.text, 'html.parser')

        summary_response = self.crawl(self.get_summary_link(URL))
        if summary_response:
            summary_soup = BeautifulSoup(summary_response.text, 'html.parser')

        reviews_response = self.crawl(self.get_review_link(URL))
        if reviews_response:
            reviews_soup = BeautifulSoup(reviews_response.text, 'html.parser')

        movie['id'] = URL.split('/')[4]
        movie['title'] = self.get_title(movie_soup)
        movie['first_page_summary'] = self.get_first_page_summary(movie_soup)
        movie['release_year'] = self.get_release_year(movie_soup)
        movie['mpaa'] = self.get_mpaa(movie_soup)
        movie['budget'] = self.get_budget(movie_soup)
        movie['gross_worldwide'] = self.get_gross_worldwide(movie_soup)
        movie['directors'] = self.get_director(movie_soup)
        movie['writers'] = self.get_writers(movie_soup)
        movie['stars'] = self.get_stars(movie_soup)
        movie['related_links'] = self.get_related_links(movie_soup)
        movie['genres'] = self.get_genres(movie_soup)
        movie['languages'] = self.get_languages(movie_soup)
        movie['countries_of_origin'] = self.get_countries_of_origin(movie_soup)
        movie['rating'] = self.get_rating(movie_soup)
        movie['summaries'] = self.get_summary(summary_soup)
        movie['synopsis'] = self.get_synopsis(summary_soup)
        movie['reviews'] = self.get_reviews_with_scores(reviews_soup)

    def get_summary_link(self, url):
        """
        Get the link to the summary page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/plotsummary is the summary page

        Parameters
        ----------
        url: str
            The URL of the site
        Returns
        ----------
        str
            The URL of the summary page
        """
        res = self.crawl(url + 'plotsummary')
        if res.status_code == 200:
            return url + 'plotsummary'
        else:
            print("failed to get summary link")

    def get_review_link(self, url):
        """
        Get the link to the review page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/reviews is the review page
        """
        res = self.crawl(url + 'reviews')
        if res.status_code == 200:
            return url + 'reviews'
        else:
            print("failed to get reviews link")

    def get_title(self, soup):
        """
        Get the title of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The title of the movie

        """
        try:
            titles = soup.find_all(class_="hero__primary-text")
            print(titles[0].string)
            return titles[0].string
        except:
            print("failed to get title")

    def get_first_page_summary(self, soup):
        """
        Get the first page summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The first page summary of the movie
        """
        try:
            attributes = {
                "role": "presentation",
                "class": "sc-466bb6c-1 dWufeH",
                "data-testid": "plot-l"
            }
            summary = soup.find_all(attrs=attributes)[0]
            return summary.string

        except:
            print("failed to get first page summary")

    def get_director(self, soup):
        """
        Get the directors of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The directors of the movie
        """

        try:
            attributes_general = {
                "role": "presentation",
                "class": "ipc-metadata-list__item",
                "data-testid": "title-pc-principal-credit"
            }
            attributes_special = {
                "role": "button",
                "class": "ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link",
                "aria-disabled": "false"
            }
            directors = (soup.find_all(attrs=attributes_general)[0]).find_all(attrs=attributes_special)
            directors_list = []
            for director in directors:
                directors_list.append(director.string)
            return directors_list
        except:
            print("failed to get director")

    def get_stars(self, soup):
        """
        Get the stars of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The stars of the movie
        """
        try:
            attributes_general = {
                "role": "presentation",
                "class": "ipc-metadata-list__item ipc-metadata-list-item--link",
                "data-testid": "title-pc-principal-credit"
            }
            attributes_special = {
                "role": "button",
                "class": "ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link",
                "aria-disabled": "false"
            }
            stars = (soup.find_all(attrs=attributes_general)[0]).find_all(attrs=attributes_special)
            stars_list = []
            for star in stars:
                stars_list.append(star.string)
            return stars_list
        except:
            print("failed to get stars")

    def get_writers(self, soup):
        """
        Get the writers of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The writers of the movie
        """
        try:
            attributes_general = {
                "role": "presentation",
                "class": "ipc-metadata-list__item",
                "data-testid": "title-pc-principal-credit"
            }
            attributes_special = {
                "role": "button",
                "class": "ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link",
                "aria-disabled": "false"
            }
            writers = (soup.find_all(attrs=attributes_general)[1]).find_all(attrs=attributes_special)
            writers_list = []
            for writer in writers:
                writers_list.append(writer.string)
            return writers_list
        except:
            print("failed to get writers")

    def get_related_links(self, soup):
        """
        Get the related links of the movie from the More like this section of the page from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The related links of the movie
        """
        try:
            attributes = {
                "class": "ipc-poster-card__title ipc-poster-card__title--clamp-2 ipc-poster-card__title--clickable"
            }
            related_links = soup.find_all(attrs=attributes)
            movie_links = []
            for related_link in related_links:
                movie_id = self.get_id_from_URL(related_link['href'])
                movie_links.append('https://www.imdb.com/title/' + movie_id + '/')
            return movie_links
        except:
            print("failed to get related links")

    def get_summary(self, soup):
        """
        Get the summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The summary of the movie
        """
        try:
            attributes_general = {
                "class": "ipc-page-section ipc-page-section--base"
            }
            attributes_special = {
                "class": "ipc-html-content-inner-div"
            }
            summaries = (soup.find_all(attrs=attributes_general)[0]).find_all(attrs=attributes_special)
            summary_texts = []
            for summary in summaries:
                summary_texts.append(summary.get_text())
            return summary_texts
        except:
            print("failed to get summary")

    def get_synopsis(self, soup):
        """
        Get the synopsis of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The synopsis of the movie
        """
        try:
            attributes_general = {
                "class": "ipc-page-section ipc-page-section--base"
            }
            attributes_special = {
                "class": "ipc-html-content-inner-div"
            }
            synopsis = (soup.find_all(attrs=attributes_general)[1]).find_all(attrs=attributes_special)
            synopsy_texts = []
            for synopsy in synopsis:
                synopsy_texts.append(synopsy.get_text())
            return synopsy_texts
        except:
            print("failed to get synopsis")

    def get_reviews_with_scores(self, soup):
        """
        Get the reviews of the movie from the soup
        reviews structure: [[review,score]]

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[List[str]]
            The reviews of the movie
        """
        try:
            attributes_review_score = {
                "class": "review-container"
            }
            attributes_review = {
                "class": "text show-more__control"
            }
            attributes_score = {
                "class": "rating-other-user-rating"
            }
            review_score_texts = []
            review_scores = soup.find_all(attrs=attributes_review_score)
            for review_score in review_scores:
                try:
                    score = review_score.find_all(attrs=attributes_score)[0].find("span").text
                except:
                    score = "No Score"
                review = review_score.find_all(attrs=attributes_review)[0].get_text()
                review_score_texts.append([review, score])
            return review_score_texts
        except:
            print("failed to get reviews")

    def get_genres(self, soup):
        """
        Get the genres of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The genres of the movie
        """
        try:
            attributes = {
                "class": "ipc-chip__text"
            }
            genres = soup.find_all(attrs=attributes)
            genres_list = []
            for genre in genres:
                genres_list.append(genre.string)
            genres_list.pop(-1)
            return genres_list
        except:
            print("Failed to get generes")

    def get_rating(self, soup):
        """
        Get the rating of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The rating of the movie
        """
        try:
            attributes = {
                "class": "sc-bde20123-1 cMEQkK"
            }
            rating = soup.find_all(attrs=attributes)[0].text
            return rating
        except:
            print("failed to get rating")

    def get_mpaa(self, soup):
        """
        Get the MPAA of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The MPAA of the movie
        """
        try:
            attributes_general = {
                "class": "sc-67fa2588-0 cFndlt"
            }
            attributes_special = {
                "role": "button",
                "class": "ipc-link ipc-link--baseAlt ipc-link--inherit-color",
                "aria-disabled": "false"
            }
            mpaa = soup.find_all(attrs=attributes_general)[0]
            mpaa = mpaa.find_all(attrs=attributes_special)[1].text
            return mpaa
        except:
            print("failed to get mpaa")

    def get_release_year(self, soup):
        """
        Get the release year of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The release year of the movie
        """
        try:
            attributes_general = {
                "class": "sc-67fa2588-0 cFndlt"
            }
            attributes_special = {
                "role": "button",
                "class": "ipc-link ipc-link--baseAlt ipc-link--inherit-color",
                "aria-disabled": "false"
            }
            release_year = soup.find_all(attrs=attributes_general)[0]
            release_year = release_year.find_all(attrs=attributes_special)[0].text
            return release_year
        except:
            print("failed to get release year")

    def get_languages(self, soup):
        """
        Get the languages of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The languages of the movie
        """
        try:
            attributes_general = {
                "data-testid": "Details",
                "class": "ipc-page-section ipc-page-section--base celwidget"
            }
            attributes = {
                "class": "ipc-metadata-list-item__content-container"
            }
            attributes_special = {
                "role": "button",
                "class": "ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link"
            }
            languages_list = []
            languages = soup.find_all(attrs=attributes_general)[0]
            languages = languages.find_all(attrs=attributes)[3]
            languages = languages.find_all(attrs=attributes_special)
            if len(languages) == 0:
                languages = (soup.find_all(attrs=attributes_general)[0]).find_all(attrs=attributes)[2]
                languages = languages.find_all(attrs=attributes_special)
            for language in languages:
                languages_list.append(language.string)
            return languages_list
        except:
            print("failed to get languages")
            return None

    def get_countries_of_origin(self, soup):
        """
        Get the countries of origin of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The countries of origin of the movie
        """
        try:
            attributes_general = {
                "data-testid": "Details",
                "class": "ipc-page-section ipc-page-section--base celwidget"
            }
            attributes = {
                "class": "ipc-metadata-list-item__content-container"
            }
            attributes_special = {
                "role": "button",
                "class": "ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link"
            }
            countries_list = []
            countries = soup.find_all(attrs=attributes_general)[0]
            countries = countries.find_all(attrs=attributes)[1]
            countries = countries.find_all(attrs=attributes_special)
            for country in countries:
                countries_list.append(country.string)
            return countries_list
        except:
            print("failed to get countries of origin")
            return None

    def get_budget(self, soup):
        """
        Get the budget of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The budget of the movie
        """
        try:
            attributes_general = {
                "data-testid": "BoxOffice",
                "class": "ipc-page-section ipc-page-section--base celwidget"
            }
            attributes = {
                "role": "presentation",
                "class": "ipc-metadata-list__item sc-1bec5ca1-2 bGsDqT"
            }
            attributes_special = {
                "class": "ipc-metadata-list-item__list-content-item"
            }
            budget = soup.find_all(attrs=attributes_general)[0]
            budget = budget.find_all(attrs=attributes)[0]
            budget = budget.find_all(attrs=attributes_special)[0].string
            return budget
        except:
            print("failed to get budget")

    def get_gross_worldwide(self, soup):
        """
        Get the gross worldwide of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The gross worldwide of the movie
        """
        try:
            attributes_general = {
                "data-testid": "BoxOffice",
                "class": "ipc-page-section ipc-page-section--base celwidget"
            }
            attributes = {
                "role": "presentation",
                "class": "ipc-metadata-list__item sc-1bec5ca1-2 bGsDqT"
            }
            attributes_special = {
                "class": "ipc-metadata-list-item__list-content-item"
            }
            gross_wordwide = soup.find_all(attrs=attributes_general)[0]
            try:
                gross_wordwide = gross_wordwide.find_all(attrs=attributes)[3]
            except:
                try:
                    gross_wordwide = gross_wordwide.find_all(attrs=attributes)[2]
                except:
                    gross_wordwide = gross_wordwide.find_all(attrs=attributes)[1]
            gross_wordwide = gross_wordwide.find_all(attrs=attributes_special)[0].string
            return gross_wordwide
        except:
            print("failed to get gross worldwide")


def main():
    imdb_crawler = IMDbCrawler(crawling_threshold=1200)
    # imdb_crawler.read_from_file_as_json()
    imdb_crawler.start_crawling()
    imdb_crawler.write_to_file_as_json()


if __name__ == '__main__':
    main()
