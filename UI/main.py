import streamlit as st
import sys
from itertools import product
import pycountry

sys.path.append('../')
from Logic import utils
import time
from enum import Enum
import random
from Logic.core.snippet import Snippet

snippet_obj = Snippet(
    number_of_words_on_each_side=5
)  # You can change this parameter, if needed.


class color(Enum):
    RED = "#FF0000"
    GREEN = "#00FF00"
    BLUE = "#0000FF"
    YELLOW = "#FFFF00"
    WHITE = "#FFFFFF"
    CYAN = "#00FFFF"
    MAGENTA = "#FF00FF"


def get_summary_with_snippet(movie_info, query):
    summary = movie_info["first_page_summary"]
    snippet, not_exist_words = snippet_obj.find_snippet(summary, query)
    if "***" in snippet:
        snippet = snippet.split()
        for i in range(len(snippet)):
            current_word = snippet[i]
            if current_word.startswith("***") and current_word.endswith("***"):
                current_word_without_star = current_word[3:-3]
                summary = summary.lower().replace(
                    current_word_without_star,
                    f"<b><font size='4' color={random.choice(list(color)).value}>{current_word_without_star}</font></b>",
                )
    return summary


def get_star_rating(rating):
    # Convert the rating to an integer number of stars (out of 5)
    rating = float(rating)
    full_stars = int(rating // 2)
    half_star = int(rating % 2 >= 1)
    empty_stars = 5 - full_stars - half_star

    # Create the star string
    stars = "★" * full_stars + "☆" * empty_stars
    if half_star:
        stars = stars[:-1] + "½"  # Use "½" for half-star representation

    return stars


def get_flag_emoji(country_name):
    try:
        country = pycountry.countries.get(name=country_name)
        if not country:
            return ""
        country_code = country.alpha_2
        flag = chr(ord(country_code[0]) + 127397) + chr(ord(country_code[1]) + 127397)
        return flag
    except:
        return ""


def search_time(start, end):
    st.success("Search took: {:.6f} milli-seconds".format((end - start) * 1e3))


def search_handling(search_button, search_term, search_max_num, search_weights, search_method, safe_search):
    if search_button:
        corrected_query = utils.correct_text(search_term, utils.movies_dataset)
        if corrected_query != search_term:
            st.warning(f"Your search terms were corrected to: {corrected_query}")
            search_term = corrected_query

        with st.spinner("Searching..."):
            time.sleep(0.5)  # for showing the spinner! (can be removed)
            start_time = time.time()
            result = utils.search(search_term, search_max_num, search_method, search_weights, safe_search)
            print(f"Result: {result}")
            end_time = time.time()
            if len(result) == 0:
                st.warning("No results found!")
                return

            search_time(start_time, end_time)

            for i in range(len(result)):
                card = st.columns([3, 1])
                info = utils.get_movie_by_id(result[i][0], utils.movies_dataset)
                with card[0].container():
                    st.title(info["title"])
                    st.markdown(f"[Link to movie]({info['URL']})")
                    st.write(f"Relevance Score: {result[i][1]}")
                    st.markdown(
                        f"<b><font size = '4'>Summary:</font></b> {get_summary_with_snippet(info, search_term)}",
                        unsafe_allow_html=True,
                    )

                with st.container():
                    if info["directors"] is not None and len(info["directors"]) > 0:
                        st.markdown("**Directors:**")
                        for director in info["directors"]:
                            st.text(director)

                with st.container():
                    if info["stars"] is not None and len(info["stars"]) > 0:
                        st.markdown("**Stars:**")
                        for star in info["stars"]:
                            st.text(star)

                    topic_card = st.columns(1)
                    with topic_card[0].container():
                        if info["genres"] is not None and len(info["genres"]) > 0:
                            st.write("Genres:")
                            for genre in info["genres"]:
                                st.markdown(
                                    f"<span style='color:{random.choice(list(color)).value}'>{genre}</span>",
                                    unsafe_allow_html=True,
                                )

                with st.expander("See More"):
                    rating = info.get('rating', 'N/A')
                    if rating != 'N/A':
                        stars = get_star_rating(rating)
                        st.markdown(f"**Rating:** {stars}")
                    else:
                        st.write("**Rating:** N/A")
                    st.write(f"**Release Year:** {info.get('release_year', 'N/A')}")
                    st.write(f"**MPAA:** {info.get('mpaa', 'N/A')}")
                    st.write(f"**Budget:** {info.get('budget', 'N/A')}")
                    st.write(f"**Gross Worldwide:** {info.get('gross_worldwide', 'N/A')}")

                    cols = st.columns(2)
                    with cols[0]:
                        if info["writers"] is not None and len(info["writers"]) > 0:
                            st.markdown("**Writers:**")
                            for writer in info["writers"]:
                                st.text(writer)

                    with cols[1]:
                        if info["countries_of_origin"] is not None and len(info["countries_of_origin"]) > 0:
                            st.markdown("**Countries of Origin:**")
                            for country in info["countries_of_origin"]:
                                flag = get_flag_emoji(country)
                                st.text(f"{flag} {country}")

                        if info["languages"] is not None and len(info["languages"]) > 0:
                            st.markdown("**Languages:**")
                            for language in info["languages"]:
                                st.text(language)

                with card[1].container():
                    try:
                        st.image('C:/Users/mehdi/IMDB-MIR/UI/images/posters/' + info["id"] + '.jpg',
                                 use_column_width=True)
                    except:
                        st.image('C:/Users/mehdi/IMDB-MIR/UI/images/posters/not_found.jpg', use_column_width=True)

                st.divider()


def main():
    col1, col2 = st.columns([3, 5])  # Adjust column widths to make the image larger on the left

    with col1:
        # Display the Alfred Hitchcock image on the leftmost side of the page
        st.image("C:/Users/mehdi/IMDB-MIR/UI/images/alfred.png", width=400)  # Adjust width as needed

    with col2:
        col2_1, col2_2 = st.columns([4, 1])
        with col2_1:
            st.title("Search Engine")
        with col2_2:
            st.image('C:/Users/mehdi/IMDB-MIR/UI/images/imdb.png', width=100)  # IMDb logo URL

        st.write(
            "This is a simple search engine for IMDB movies. You can search through IMDB dataset and find the most "
            "relevant movie to your search terms."
        )
        st.markdown(
            '<span style="color:yellow">Developed By: MIR Team at Sharif University</span>',
            unsafe_allow_html=True,
        )

        search_term = st.text_input("Search Term")
        with st.expander("Advanced Search"):
            search_max_num = st.number_input(
                "Maximum number of results", min_value=5, max_value=100, value=10, step=5
            )
            weight_stars = st.slider(
                "Weight of stars in search",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.1,
            )

            weight_genres = st.slider(
                "Weight of genres in search",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.1,
            )

            weight_summary = st.slider(
                "Weight of summary in search",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.1,
            )

            safe_search = st.checkbox("Safe Search", value=True)

            search_weights = [weight_stars, weight_genres, weight_summary]

            search_method = st.selectbox(
                "Search method",
                ["OkapiBM25", "Custom"]
            )

            if search_method == "Custom":
                l_options = ["n", "l"]
                t_options = ["n", "t"]
                c_options = ["n", "c"]

                custom_method = ""
                custom_method += st.selectbox("Choose query first position method", l_options)
                custom_method += st.selectbox("Choose query second position method", t_options)
                custom_method += st.selectbox("Choose query third position method", c_options)
                custom_method += "."
                custom_method += st.selectbox("Choose documents first position method", l_options)
                custom_method += st.selectbox("Choose documents second position method", t_options)
                custom_method += st.selectbox("Choose documents third position method", c_options)

                search_method = custom_method

        search_button = st.button("Search!")

        search_handling(
            search_button,
            search_term,
            search_max_num,
            search_weights,
            search_method,
            safe_search,
        )


if __name__ == "__main__":
    main()
