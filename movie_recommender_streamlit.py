import streamlit as st
import pandas as pd
import numpy as np
import requests
from surprise import Reader, Dataset, KNNBasic, accuracy, SVD
from streamlit_js_eval import streamlit_js_eval


# page configuration

st.set_page_config(page_title="WBSFLIX",
                   page_icon="🎬", layout = 'wide')


tabs_font_css = """
<style>
div[class*="stTextInput"] label p {
  font-size: 26px;
}
div[class*="stSelectbox"] label p {
  font-size: 26px;
}
div[class*="stRadio"] label p {
  font-size: 26px;
}
</style>
"""

# Apply the CSS styling
st.write(tabs_font_css, unsafe_allow_html=True)


st.image("https://www.3playmedia.com/wp-content/uploads/sound-effects-blog-header.jpg", width = 800)
st.title('WBSFlix')
st.write("""
         ### 🎬 Welcome to the next generation movie recommendation app
         
         """)
# general df stuff

movies = pd.read_csv('movies_cl.csv')
tags = pd.read_csv('tags.csv')
ratings = pd.read_csv('ratings.csv')
links = pd.read_csv('links.csv')

links = links.dropna()
links['tmdbId'] = links['tmdbId'].astype(int)

movies_ratings = ratings.merge(movies, how= 'left', on= 'movieId')




def images_url(df):
        images_list = []
        for i, row in df.iterrows():
            
            tmdbId = row['tmdbId']
            url = f'https://api.themoviedb.org/3/movie/{tmdbId}/images?include_image_language=en'

            headers = {
                "accept": "application/json",
                "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJlMjg5NDc2NmVlYmIzNmNlMjUwZTgyZTc3NzFjYTc3NSIsInN1YiI6IjY0Zjc0MGU2ZjI5ZDY2MzU2NTQ2MGUzNiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.v6ThpjFhLZ2iap3F-7zU1-4BQFxVcutnKI_EDGPiEGA"
            }

            response = requests.get(url, headers=headers)
            json = response.json()

            if 'posters' in json:
                file_path = json['posters'][0]['file_path']
                url_image = f'https://image.tmdb.org/t/p/original{file_path}'
                images_list.append(url_image)
            else:
                images_list.append('https://img.freepik.com/premium-vector/cinema-movie-background-popcorn-filmstrip-clapboard-tickets-movie-time-background_41737-248.jpg')
        return images_list


#popular ranking
def popular_ranking_outer():
    st.subheader('Popular Movies :100:')


    def pop_ranking_recommendation(number):
        rating_count_df = movies_ratings.groupby(['movieId', 'title'])['rating'].agg(['mean', 'count']).reset_index()

        #calculate the mean and count normalization (== MinMaxScaler)
        rating_count_df['mean_normalized'] = (rating_count_df['mean'] - rating_count_df['mean'].min()) / (rating_count_df['mean'].max() - rating_count_df['mean'].min())
        rating_count_df['count_normalized'] = (rating_count_df['count'] - rating_count_df['count'].min()) / (rating_count_df['count'].max() - rating_count_df['count'].min())

        #define a weight for the mean and count
        mean_weight = 0.7
        count_weight = 0.3

        #calculate the combined metric
        rating_count_df['combined_metric'] = mean_weight * rating_count_df['mean_normalized'] + count_weight * rating_count_df['count_normalized']

        sorted_movies = rating_count_df.sort_values(by='combined_metric', ascending=False).head(number)
        return sorted_movies

    number_of_recommendations = 10
    popular_movies = pop_ranking_recommendation(number_of_recommendations)

    # get images from popular movies recommendations for slider

    get_tmbdId_popular = popular_movies.merge(links, how = 'left', on= 'movieId')

    images_popular_movies = images_url(get_tmbdId_popular)

    title_list = popular_movies['title'].tolist()

    result_popular = st.image(images_popular_movies, width=130, caption=title_list)
    return result_popular
    
    



# item-based recommendations

def item_based_recommendations(movie_title):


    st.subheader(f'If you enjoyed watching {movie_title}, you will also like these movies :film_projector:')

    user_movies_matrix = pd.pivot_table(data=movies_ratings,
                                    values='rating',
                                    index='userId',
                                    columns='title',
                                    fill_value=0)

    from sklearn.metrics.pairwise import cosine_similarity

    cosines_matrix = pd.DataFrame(cosine_similarity(user_movies_matrix.T),
                                        columns=user_movies_matrix.columns,
                                        index=user_movies_matrix.columns)

    def recommendation_cosine(movie_title, number=10):
        cosines_df = pd.DataFrame(cosines_matrix[movie_title])

        # Rename the column
        cosines_df = cosines_df.rename(columns={movie_title: 'movies_cosine'})

        # Remove the row with the index
        cosines_df = cosines_df[cosines_df.index != movie_title]


        cosines_df = cosines_df.sort_values(by="movies_cosine", ascending=False)

        # Find out the number of users rated the movie we are looking for and the other movie
        no_of_users_rated_both_movies = [sum((user_movies_matrix[title] > 0) & (user_movies_matrix[movie_title] > 0)) for title in cosines_df.index]

        # Create a column for the number of users who rated the two movies
        cosines_df['users_who_rated_both_movies'] = no_of_users_rated_both_movies


        # Optional: Remove recommendations that have less than X users who rated both movies
        cosines_df = cosines_df[cosines_df["users_who_rated_both_movies"] >=1]

        # Display the first 10 rows of the DataFrame

        top_n_cosine = (cosines_df.head(number)
                        .merge(movies, how = 'left', on = 'title')
                                    .reset_index()
                                    )
        return top_n_cosine
    


    item_based_recommendations = recommendation_cosine(movie_title)

    #display posters of the best item based recommendations
    get_tmbdId_item_based = item_based_recommendations.merge(links, how = 'left', on= 'movieId')
    images_item_based = images_url(get_tmbdId_item_based)
    title_list2 = item_based_recommendations['title'].tolist()


    result_item_based = st.image(images_item_based, width=130, caption=title_list2)
    return result_item_based


# user based recommendations
def user_based_recommendations(movie_title):
    st.subheader('Other Moviefreaks also liked...')
    highest_rating = movies_ratings[movies_ratings['title'] == movie_title]['rating'].max()

    # Step 3: Filter data for the target movie and the highest rating
    user_profile = movies_ratings[(movies_ratings['title'] == movie_title) &
                                (movies_ratings['rating'] == highest_rating)]['userId'].tolist()

    target_user_list = []

    # Iterate through user IDs in the user_profile
    for user in user_profile:
        # Filter movies_ratings for the current user
        target_users = movies_ratings.loc[movies_ratings['userId'] == user]
        
        #append the user-specific dataframe to the list
        target_user_list.append(target_users)
        combined_user_df = pd.concat(target_user_list, ignore_index=True)
        movie_ratings_count = combined_user_df.groupby(['title', 'movieId'])['rating'].count().reset_index()
        top_rated_movies = movie_ratings_count.sort_values(by='rating', ascending=False)
        top_ten_rated_movies = top_rated_movies.head(10)
    user_based_recommendations = top_ten_rated_movies.merge(links, how = 'left', on = 'movieId') 
    images_user_based = images_url(user_based_recommendations)  
    title_list_user = user_based_recommendations['title'].tolist()

    result_user_based = st.image(images_user_based, width=130, caption=title_list_user)
    
    return result_user_based





# genre based recommendations

def genre_based_recommendations(movie_title):
    st.subheader(f'Genre based recommendations for {movie_title}')

    def genre_based_recommendation(movie_title):

        #get the genres of the target movie
        target_genres = movies.loc[movies['title'] == movie_title, 'genres'].values[0].split('|')

        #get exact matches
        exact_matches = movies[movies['genres'] == '|'.join(target_genres)]

        if len(exact_matches) < 10:
            one_less_genre = movies[movies['genres'].apply(lambda x: set(x.split('|')).issubset(set(target_genres[:-1])))]
            
            #exclude target movie
            one_less_genre = one_less_genre[one_less_genre['title'] != movie_title]
            count_exact_matches = 10 - len(exact_matches)
            
            if len(one_less_genre) > count_exact_matches:
                target_year = movies.loc[movies['title'] == movie_title, 'year_of_publication'].values[0]
                one_less_genre['year_difference'] = abs(one_less_genre['year_of_publication'] - target_year)
                one_less_genre.sort_values(by = 'year_difference')
                one_less_genre.head(count_exact_matches)
            
                
            
                recommendations = pd.concat([exact_matches, one_less_genre])
        else:
            recommendations = exact_matches


        top_10_recommendations = recommendations.head(10)

        return top_10_recommendations

    genre_based_recommendations = genre_based_recommendation(movie_title)

    #st.dataframe (genre_based_recommendations)

    get_tmbdId_genre = genre_based_recommendations.merge(links, how = 'left', on= 'movieId')

    images_genre_movies = images_url(get_tmbdId_genre)

    title_list_genre = genre_based_recommendations['title'].tolist()


    result_genre_based = st.image(images_genre_movies, width=130, caption=title_list_genre)
    return result_genre_based


#chatbot 

from fuzzywuzzy import fuzz

#get a similarity score with the fuzzywuzzy library to check for similar movie titles


def find_similar_movies(input_movie, df, threshold=75):
    similar_movies = []
    
    for index, row in df.iterrows():
        movie_title = row['title']
        similarity_score = fuzz.partial_ratio(input_movie.lower(), movie_title.lower())
        
        if similarity_score >= threshold:
            similar_movies.append((movie_title, similarity_score))
    
    similar_movies_df = pd.DataFrame(similar_movies).rename(columns = {0 : 'title', 1 : 'similarity_score'}).sort_values(by = 'similarity_score', ascending = False).reset_index()
    similar_movies_df.drop('index', inplace = True, axis = 1)
    
    return similar_movies_df



#chatbot structure


similar_movies_df = pd.DataFrame()
movie_select_count = 0
movie_input_counter = 0
movie_radio_counter = 0

if 'button' not in st.session_state:
    st.session_state['button'] = 0


users_name = st.text_input('What is your name?')

if users_name:
    users_name = users_name.title()
    st.write(f'Hi {users_name}! :wave: Nice to meet you.')
    conversation_active = True

    while conversation_active:
        movie_input_counter += 1 
        favorite_movie = st.text_input(f'What is your favorite movie? :film_frames:', key=f'movie_input_{movie_input_counter}')

        if favorite_movie:
            favorite_movie = favorite_movie.title()
            try:
                similar_movies_df = find_similar_movies(favorite_movie, movies)
            except:
                movie_radio_counter += 1
                yes_no = st.radio('No similar movies found. Would you want to enter another movie title?', ['Yes!', 'No!'], key=f'radio_{movie_radio_counter}')
            
                if yes_no == 'Yes!':
                    st.write("You selected 'Yes!' :tada:")
                    conversation_active = True
                else:
                    st.write("You selected 'No!' Hope to see you again next time! :rainbow:")
                    conversation_active = False
                    restart = st.button('Restart', key=f'button_{st.session_state.button}')
                    if restart:          
                            streamlit_js_eval(js_expressions="parent.window.location.reload()")
                
            else:
                if not similar_movies_df.empty:
                    similar_movies_100 = similar_movies_df.loc[similar_movies_df['similarity_score'] >= 90]
                    #st.write(f"These are the most similar movies we found to {favorite_movie}")

                    if len(similar_movies_100) > 1:
                        movie_select_count += 1
                        selected_movie = st.selectbox('We found a bunch of corresponding movies. Please select a movie:', similar_movies_100['title'].tolist(), key=f'selected_movie_{movie_select_count}')
                                               
                        movie_title = selected_movie
                        item_based = item_based_recommendations(movie_title)
                        st.write("".join("🍿 " for _ in range(64)))                         
                        user_based= user_based_recommendations(movie_title)
                        
                        st.write("".join("🍿 " for _ in range(64)))                         
                        genre_based = genre_based_recommendations(movie_title)
                        st.write("".join("🍿 " for _ in range(64)))
                        popular_ranking = popular_ranking_outer()

                        st.balloons()
                        
                        movie_radio_counter += 1
                        yes_no = st.radio('Do you want to check out the recommendations also for a different movie?', ['Yes!', 'No!'], key=f'radio_{movie_radio_counter}')
                    
                        if yes_no == 'Yes!':
                            st.write("You selected 'Yes!' :tada:")
                            conversation_active = True
                        else:
                            st.write("You selected 'No!' Hope to see you again next time!:rainbow:")
                            conversation_active = False
                            restart = st.button('Restart', key=f'button_{st.session_state.button}')
                            if restart:
                                streamlit_js_eval(js_expressions="parent.window.location.reload()")
                                               
                    elif len(similar_movies_100) == 1:
                        movie_100 = similar_movies_100['title'].iloc[0]
                        movie_radio_counter += 1
                        yes_no = st.radio(f'Is {movie_100} your favorite movie?', ['Yes!', 'No!'], key=f'radio_{movie_radio_counter}')
                        
                        
                        if yes_no == 'No!':
                            movie_radio_counter += 1
                            yes_no = st.radio('Would you want to enter another movie title?', ['Yes!', 'No!'], key=f'radio_{movie_radio_counter}')
                        
                            if yes_no == 'Yes!':
                                st.write("You selected 'Yes!' :tada:")
                                conversation_active = True
                            else:
                                st.write("You selected 'No!' Hope to see you again next time!:rainbow:")
                                conversation_active = False
                               
                                restart = st.button('Restart', key=f'button_{st.session_state.button}')
                                if restart:
                                    streamlit_js_eval(js_expressions="parent.window.location.reload()")
                            
                        else:
                            st.write('Great! Then here are your personal movie recommendations:')
                            
                            movie_title = movie_100
                            item_based = item_based_recommendations(movie_title)
                            st.write("".join("🍿 " for _ in range(64)))
                            user_based= user_based_recommendations(movie_title)
                            st.write("".join("🍿 " for _ in range(64)))
                            genre_based = genre_based_recommendations(movie_title)
                            st.write("".join("🍿 " for _ in range(64)))
                            popular_ranking = popular_ranking_outer()
                            st.balloons()
                            movie_radio_counter += 1
                            yes_no = st.radio('Do you want to check out the recommendations also for a different movie?', ['Yes!', 'No!'], key=f'radio_{movie_radio_counter}')
                        
                            if yes_no == 'Yes!':
                                st.write("You selected 'Yes!' :tada:")
                                conversation_active = True
                            else:
                                st.write("You selected 'No!' Hope to see you again next time!:rainbow:")
                                conversation_active = False
                                
                                restart = st.button('Restart', key=f'button_{st.session_state.button}')
                                if restart:
                                    streamlit_js_eval(js_expressions="parent.window.location.reload()")
                    else:
                        movie_radio_counter += 1
                        yes_no = st.radio('No similar movies found. Would you want to enter another movie title?', ['Yes!', 'No!'], key=f'radio_{movie_radio_counter}')
                    
                        if yes_no == 'Yes!':
                            st.write("You selected 'Yes!' :tada:")
                            conversation_active = True
                        else:
                            st.write("You selected 'No!' Hope to see you again next time!:rainbow:")
                            conversation_active = False
                            
                            restart = st.button('Restart', key=f'button_{st.session_state.button}')
                            if restart:
                                streamlit_js_eval(js_expressions="parent.window.location.reload()")
        else:
            conversation_active = False
