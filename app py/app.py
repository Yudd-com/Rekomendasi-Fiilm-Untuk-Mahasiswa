from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Muat data film
movies_df = pd.read_csv('E:\Website Film Tubes PASD\Folder Gemini\data\movies.csv')  # Pastikan path sesuai struktur foldermu

# Ambil semua genre unik
def extract_genres(movies_df):
    genres_set = set()
    for genres_str in movies_df['genres']:
        # Parsing genre dari string JSON-like
        import ast
        try:
            genres_list = ast.literal_eval(genres_str)
            for g in genres_list:
                genres_set.add(g['name'])
        except:
            continue
    return sorted(list(genres_set))

@app.route('/', methods=['GET'])
def index():
    genres = extract_genres(movies_df)
    return render_template('index.html', genres=genres)

@app.route('/recommend', methods=['POST'])
def recommend():
    tahun = request.form.get('tahun')  # Ambil tahun dari input user
    genre = request.form.get('genre')

    filtered = movies_df.copy()
    if tahun:
        # Ambil 4 digit pertama dari release_date (tahun)
        filtered = filtered[filtered['release_date'].fillna('').str[:4] == str(tahun)]
    if genre:
        import ast
        def has_genre(genres_str):
            try:
                genres_list = ast.literal_eval(genres_str)
                return any(g['name'] == genre for g in genres_list)
            except:
                return False
        filtered = filtered[filtered['genres'].apply(has_genre)]

    movies = []
    for _, row in filtered.iterrows():
        movies.append({
            'title': row['title'],
            'genres': row['genres'],
            'release_date': row['release_date'],
            'why_recommended': f"Film dari tahun {tahun} dan genre {genre}" if tahun or genre else "",
            'poster_url': row.get('poster_url', None)
        })

    return render_template('recommendations.html', movies=movies)


# ...existing code...

if __name__ == '__main__':
    app.run(debug=True)