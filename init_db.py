import os
import psycopg2

# open a connection to the flask_db database
conn = psycopg2.connect(
    host="localhost",
    port="5435", # added, original default is 5432
    database="flask_db",
    user=os.environ['DB_USERNAME'],
    password=os.environ['DB_PASSWORD']
)

# create and open a cursor to perform database operations
# allows Python code to execute PostgreSQL commands in a database session
cur = conn.cursor()

# execute commands to create a new table
# cur.execute('DROP TABLE IF EXISTS music;')
cur.execute('CREATE TABLE music (id serial PRIMARY KEY,'
                      	'song VARCHAR(255) NOT NULL,'
                       	'artist VARCHAR(255) NOT NULL,'
                        'album VARCHAR(255) NOT NULL,'
                        'duration_min DECIMAL(10,2) NOT NULL,'
                        'streams BIGINT NOT NULL,'
                        'image VARCHAR(255) NOT NULL);'
)

# insert data into the table
cur.execute('INSERT INTO music (song, artist, album, duration_min, streams, image)'
            'VALUES (%s, %s, %s, %s, %s, %s)',
            ('Road to the West',
             'SEATBELTS',
             'COWBOY BEBOP (Original Motion Picture Soundtrack 3 - Blue)',
             2.9167,
             10333057,
             'https://i.scdn.co/image/ab67616d0000b2731d4aac2fbcf95a39353fb846')
            )

cur.execute('INSERT INTO music (song, artist, album, duration_min, streams, image)'
            'VALUES (%s, %s, %s, %s, %s, %s)',
            ('Nutshell',
             'Alice In Chains',
             'Jar Of Flies',
             4.3167,
	         396394446,
             'https://i.scdn.co/image/ab67616d0000b27325b42be683b8d3c6500db726')
            )

# commit the transaction and apply the changes to the database
conn.commit()

# clean by closing cursor and connection
cur.close()
conn.close()
