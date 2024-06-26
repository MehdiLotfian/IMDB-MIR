import json

import requests
from bs4 import BeautifulSoup
import os


def download_posters(movie_list):
    # Ensure the directory to save posters exists
    if not os.path.exists('posters'):
        os.makedirs('posters')

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for movie in movie_list:
        imdb_url = "https://imdb.com/title/" + movie['id']
        movie_id = movie['id']

        # Fetch the IMDb page
        response = requests.get(imdb_url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to retrieve {imdb_url}")
            continue

        # Parse the IMDb page
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the poster image URL
        poster_tag = soup.find('div', class_='ipc-poster')
        if poster_tag is None:
            print(f"No poster found for {imdb_url}")
            continue

        img_tag = poster_tag.find('img')
        if img_tag is None or 'src' not in img_tag.attrs:
            print(f"No image tag found for {imdb_url}")
            continue

        poster_url = img_tag['src']

        # Download the poster image
        img_response = requests.get(poster_url, headers=headers)
        if img_response.status_code != 200:
            print(f"Failed to download image from {poster_url}")
            continue

        # Save the image with the movie ID as the filename
        img_filename = f"posters/{movie_id}.jpg"
        with open(img_filename, 'wb') as img_file:
            img_file.write(img_response.content)

        print(f"Downloaded poster for {movie_id}")


# List of IMDb movie URLs and IDs
movies = json.load(open("C:/Users/mehdi/IMDB-MIR/Logic/core/IMDB_crawled.json"))

download_posters(movies)
