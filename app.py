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
import numpy as np
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
    return redirect(url_for('index'))

@app.route('/recommend/', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        # get music from database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT song, artist, album, duration_min, streams FROM music;')
        # cur.execute('SELECT * FROM music;') if image link too
        music = cur.fetchall()
        cur.close()
        conn.close()

    # in an if (): return render_template('recommend.html', recommendations=recommendations)
    
    return render_template('recommend.html')



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
