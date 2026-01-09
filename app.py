from flask import Flask, render_template, request, url_for, redirect
from dotenv import load_dotenv
import os
import psycopg2

# for Spotify API
import base64
from requests import post, get
import json

# for recommender
import pandas as pd
# import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# import matplotlib.pyplot as plt
# import seaborn as sns


load_dotenv()

# for Spotify API
client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
# print(client_id, client_secret)

app = Flask(__name__)

# Global variables for recommendation system
kaggle_data = None  # Full Kaggle dataset
user_data = None    # Your database songs
cosine_sim = None

# set up your database connection and create a single Flask route to use that connection
def get_db_connection():
    # opens connection
    conn = psycopg2.connect(host='localhost',
                            port=5435,
                            database='flask_db',
                            user=os.environ['DB_USERNAME'],         # reads from .env
                            password=os.environ['DB_PASSWORD'])
    return conn
    # connection object used to access the database


# get Spotify dataset from Kaggle
# load once upon startup
def load_large_dataset():
    global kaggle_data

    try:
        kaggle_data = pd.read_csv('spotify_dataset.csv')
        print(f"✓ Loaded {len(kaggle_data)} songs from Kaggle dataset")

        # combine features into a single feature
        kaggle_data['combined_features'] = (
            kaggle_data['track_name'].fillna('') + ' ' +
            kaggle_data['artists'].fillna('') + ' ' +
            kaggle_data['album_name'].fillna('')
        )

        return True
    
    except FileNotFoundError:
        print("spotify_dataset.csv not found!!")
        print("https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset")
        return False
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
    
    
# load and build
    # 1. Reload songs from database (gets the NEW song)
    # 2. Combine with Kaggle dataset
    # 3. Recalculate TF-IDF matrix
    # 4. Rebuild cosine_sim matrix (now 4x4 with Song D!)
def recommender():
    # combine my songs and the Kaggle dataset to build the similarity matrix
    global user_data, kaggle_data, cosine_sim

    # load my songs from my database
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM music;')
    data = cur.fetchall()
    cur.close()
    conn.close()

    # Convert your songs to DataFrame
    user_data = pd.DataFrame(data, columns=['id', 'song', 'artist', 'album', 'genre', 'duration_min', 'streams', 'image', 'release_date', 'popularity'])
    
    if len(user_data) == 0:
        print("No songs in personal database yet")
        return False
    
    # Convert genre list to string for your songs
    user_data['genre_str'] = user_data['genre'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
    
    # Create combined features for your songs
    user_data['combined_features'] = (
        user_data['genre_str'].fillna('') + ' ' +
        user_data['artist'].fillna('') + ' ' +
        user_data['song'].fillna('')
    )
    
    # Add a marker to distinguish your songs from Kaggle songs
    user_data['source'] = 'user_library'
    
    if kaggle_data is not None:
        # Add marker to Kaggle songs
        kaggle_subset = kaggle_data.copy()
        kaggle_subset['source'] = 'kaggle'
        
        # Combine both datasets
        # We only need combined_features and source columns for recommendation
        user_features = user_data[['combined_features', 'source']].copy()
        kaggle_features = kaggle_subset[['combined_features', 'source']].copy()
        
        combined_data = pd.concat([user_features, kaggle_features], ignore_index=True)
        
        print(f"✓ Combined dataset: {len(user_data)} your songs + {len(kaggle_data)} Kaggle songs")
    else:
        # If Kaggle dataset not loaded, just use your songs
        combined_data = user_data[['combined_features', 'source']].copy()
        print("⚠ Using only your library (Kaggle dataset not loaded)")
    
    # Build TF-IDF matrix on combined dataset
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(combined_data['combined_features'])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    print("✓ Recommendation system ready!")
    return True


def get_recommendations(song_id, top_n=10, include_user_songs=False):
    """
    Get recommendations from Kaggle dataset based on your song
    
    Args:
        song_id: ID of song from your database
        top_n: Number of recommendations to return
        include_user_songs: If True, can recommend from your library too
    
    Returns:
        List of recommended songs from Kaggle dataset
    """
    global user_data, kaggle_data, cosine_sim
    
    if cosine_sim is None or user_data is None:
        return []
    
    # Find the index of your song in the combined dataset
    # Your songs are at the beginning (indices 0 to len(user_data)-1)
    song_row = user_data[user_data['id'] == song_id]
    if len(song_row) == 0:
        print("Song not found")
        return []
    
    # Get the position in user_data, which is same as position in combined dataset
    idx = user_data[user_data['id'] == song_id].index[0]
    
    # Get similarity scores for all songs
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Filter recommendations
    recommendations = []
    user_library_size = len(user_data)
    
    for i, score in sim_scores:
        # Skip the song itself
        if i == idx:
            continue
        
        # Check if this is from Kaggle dataset
        is_kaggle_song = i >= user_library_size
        
        if include_user_songs:
            # Include everything
            recommendations.append((i, score, is_kaggle_song))
        else:
            # Only include Kaggle songs
            if is_kaggle_song:
                recommendations.append((i, score, is_kaggle_song))
        
        # Stop when we have enough
        if len(recommendations) >= top_n:
            break
    
    # Convert to list of song details
    result = []
    for idx_val, score, is_kaggle in recommendations:
        if is_kaggle:
            # Get from Kaggle dataset
            kaggle_idx = idx_val - user_library_size
            row = kaggle_data.iloc[kaggle_idx]
            result.append({
                'song': row['track_name'],
                'artist': row['artists'],
                'album': row.get('album_name', 'Unknown'),
                'genre': row.get('track_genre', 'Unknown'),
                'popularity': row.get('popularity', 0),
                'duration_min': row.get('duration_ms', 0) / 60000,
                'source': 'kaggle',
                'similarity_score': round(score * 100, 1)
            })
        else:
            # Get from your library
            row = user_data.iloc[idx_val]
            result.append({
                'song': row['song'],
                'artist': row['artist'],
                'album': row['album'],
                'genre': row['genre_str'] if 'genre_str' in row else row['genre'],
                'popularity': row.get('popularity', 0),
                'duration_min': row['duration_min'],
                'image': row.get('image', ''),
                'source': 'user_library',
                'similarity_score': round(score * 100, 1)
            })
    
    return result




# index page to display with main route
@app.route('/')
def index():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM music;')
    music = cur.fetchall() # save data selected into songs variable
    cur.close()
    conn.close()
    return render_template('index.html', music=music)

# route for adding new music
# GET requests are used to retrieve data from the server.
# POST requests are used to post data to a specific route.
@app.route('/create/', methods=('GET', 'POST'))
def create():
    # extract the data that the user submits from the request.form object
    if request.method == 'POST':
        song = request.form['song']
        artist = request.form['artist']
        streams = int(request.form['streams'])
        image_url = request.form['link']

        # use to get track info
        id = get_song_id(song, artist)
        if (id == None):
            return None
        json = song_search(id)
        genres = get_genre(id)
        
        album = json["album"]["name"]
        duration = float(json["duration_ms"]) / 60000       # convert from ms to minutes
        if (image_url == ""):
            image_url = json["album"]["images"][0]["url"]   # get size 600 pixels
        release_date = json["album"]["release_date"]
        popularity = json["popularity"]

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('INSERT INTO music (song, artist, album, duration_min, streams, image, genre, release_date, popularity)'
                    'VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)',
                    (song, artist, album, duration, streams, image_url, genres, release_date, popularity)
        )
        
        conn.commit()
        cur.close()
        conn.close()

        # rebuild recommendation after new added songs
        recommender()

        return redirect(url_for('index')) # so the user can see the new addition directly after

    return render_template('create.html')

@app.route('/update', methods=['POST'])
def update():
    conn = get_db_connection()
    cur = conn.cursor()

    # update attributes
    # Get the data from the form
    song = request.form['song']
    artist = request.form['artist']
    id = request.form['id']

    # Update the data in the table
    cur.execute('UPDATE music SET song=%s, artist=%s WHERE id=%s', (song, artist, id))

    conn.commit()
    cur.close()
    conn.close()

    recommender()

    return redirect(url_for('index'))


@app.route('/delete', methods=['POST'])
def delete():
    conn = get_db_connection()
    cur = conn.cursor()

    # get the data 
    id = request.form['id']

    # delete entries
    # cur.execute('''DELETE FROM products WHERE id=%s''', (id,))
    cur.execute('DELETE FROM music WHERE id=%s', (id,))

    conn.commit()
    cur.close()
    conn.close()

    # keeps it up to date
    # slower with a larger Kaggle dataset
    # could remove if I don't need to update and wanted faster CRUD
    # requires restarting flask app for new recs
    recommender()

    return redirect(url_for('index'))

@app.route('/recommend/', methods=['GET', 'POST'])
def recommend():
    recommendations = []
    selected_song = None
    all_songs = []
    
    # Get all songs for dropdown
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT id, song, artist FROM music ORDER BY song;')
    all_songs = cur.fetchall()
    cur.close()
    conn.close()


    if request.method == 'POST':
        song_id = int(request.form.get('song_id'))
        
        # Get selected song details
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT * FROM music WHERE id=%s;', (song_id,))
        selected_song = cur.fetchone()
        cur.close()
        conn.close()
        
        # Get recommendations from Kaggle dataset
        recommendations = get_recommendations(song_id, top_n=20, include_user_songs=False)
    
    return render_template('recommend.html', 
                         recommendations=recommendations, 
                         selected_song=selected_song,
                         all_songs=all_songs)



# Spotify API functions

def get_token():
    # create authorization string and encode with base 64
    # concatenate client_id and client_secret, then encode
    # send to receieve authorization token
    # request access token valid for an hour, send POST request

    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}     # HTTP body 
    result = post(url, headers=headers, data=data)  # POST request to the token endpoint URI
    json_result = json.loads(result.content)
    token = json_result["access_token"]
    return token

# construct header needed for sending future requests
def get_auth_header(token):
    return {"Authorization": "Bearer " + token}

def get_song_id(song, artist):
    # get token
    token = get_token()
    headers = get_auth_header(token)
    # query parameters
    url = "https://api.spotify.com/v1/search"
    query_url = f"https://api.spotify.com/v1/search?q={song}&type=track&limit=3"
    params = {"q": song, "type": "track", "limit": 3}
    result = get(url, headers=headers, params=params)
    json_results = json.loads(result.content)["tracks"]["items"]
    # iterate to match to artist name
    if len(json_results) == 0:
        print("No results")
        # redirect to failure page or pop up
        return None
    
    id = json_results[0]["id"]

    if (artist):
        for x in range(3):
            if (artist == json_results[x]["artists"][0]["name"]):   # .lower()
                id = json_results[x]["id"]
                break
    
    return id

# search for song
def song_search(id):
    # get token
    token = get_token()
    headers = get_auth_header(token)
    query_url = f"https://api.spotify.com/v1/tracks/{id}"
    result = get(query_url, headers=headers)
    json_result = json.loads(result.content)

    return json_result

# get list of genres associated with the artist
def get_genre(id):
    token = get_token()
    headers = get_auth_header(token)
    # query parameters
    artist_id = song_search(id)["artists"][0]["id"]
    url = f"https://api.spotify.com/v1/artists/{artist_id}"
    result = get(url, headers=headers)
    genres = json.loads(result.content)["genres"]

    return genres




# Initialize recommendation system on startup
print("=" * 60)
print("INITIALIZING MUSIC RECOMMENDATION SYSTEM")
print("=" * 60)

# Step 1: Load Kaggle dataset
if load_large_dataset():
    print("✓ Step 1: Kaggle dataset loaded")
else:
    print("⚠ Step 1: Kaggle dataset not available")
    print("  Download: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset")
    print("  Save as: dataset.csv in your project folder")

# Step 2: Build recommender
try:
    if recommender():
        print("✓ Step 2: Recommendation system ready!")
    else:
        print("⚠ Step 2: Add songs to your database first")
except Exception as e:
    print(f"⚠ Error: {e}")

print("=" * 60)

# issues with flask run -> flask run --debug --no-reload --port 5001
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5001)