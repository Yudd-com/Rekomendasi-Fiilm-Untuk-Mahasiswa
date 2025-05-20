dummy_movie_ids = movies_df['movieId'].sample(num_recommendations).tolist() if movies_df is not None elsedummy_movie_ids = movies_df['movieId'].sample(num_recommendations).tolist() if movies_df is not None elsedummy_movie_ids = movies_df['movieId'].sample(num_recommendations).tolist() if movies_df is not None elseimport pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel # Mirip dengan cosine_similarity, sering digunakan dengan TF-IDF
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from collections import defaultdict
import pickle # Untuk menyimpan model

# --- 1. Memuat dan Memproses Data ---
def load_data(E:\Website Film Tubes PASD\Folder Gemini\data\movies.csv):
    try:
        movies_df = pd.read_csv(movies_csv_path)
        ratings_df = pd.read_csv(ratings_csv_path)
        print("Data berhasil dimuat.")
    except FileNotFoundError:
        print(f"Error: Pastikan file CSV ada di path yang benar ('{movies_csv_path}' dan '{ratings_csv_path}').")
        return None, None

    # Membersihkan data film (contoh sederhana)
    # Anda mungkin perlu penanganan missing values yang lebih canggih
    movies_df['overview'] = movies_df['overview'].fillna('')
    movies_df['genres'] = movies_df['genres'].fillna('')
    
    # Membuat kolom 'soup' untuk content-based filtering
    # Menggabungkan beberapa fitur teks menjadi satu string untuk analisis
    # Sesuaikan kolom ini berdasarkan data Anda (misalnya, keywords, cast, director)
    movies_df['soup'] = movies_df['overview'] + ' ' + movies_df['genres'] 
    print(f"Jumlah film: {len(movies_df)}, Jumlah rating: {len(ratings_df)}")
    return movies_df, ratings_df

# --- 2. Content-Based Filtering ---
def train_content_based_model(movies_df):
    """Melatih model content-based filtering menggunakan TF-IDF."""
    if movies_df is None:
        return None, None

    print("\nMelatih model Content-Based Filtering...")
    # Inisialisasi TF-IDF Vectorizer. Stop words adalah kata-kata umum yang diabaikan.
    tfidf = TfidfVectorizer(stop_words='english') # [2, 3]

    # Membuat matriks TF-IDF dari kolom 'soup'
    tfidf_matrix = tfidf.fit_transform(movies_df['soup']) # [2, 3]

    # Menghitung cosine similarity matrix
    # linear_kernel lebih efisien untuk matriks TF-IDF
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix) # [2, 3]
    print("Model Content-Based Filtering selesai dilatih.")
    
    # Menyimpan model TF-IDF dan matriks kesamaan
    try:
        with open('models/content_vectorizer.pkl', 'wb') as f_vec:
            pickle.dump(tfidf, f_vec) # [4]
        with open('models/content_similarity_matrix.pkl', 'wb') as f_sim:
            pickle.dump(cosine_sim, f_sim) # [4]
        print("Model Content-Based (TF-IDF & Similarity Matrix) berhasil disimpan.")
    except Exception as e:
        print(f"Gagal menyimpan model Content-Based: {e}")
        
    return tfidf, cosine_sim

def get_content_based_recommendations(movie_title, movies_df, cosine_sim, top_n=10):
    """Mendapatkan rekomendasi film berdasarkan kemiripan konten."""
    if movies_df is None or cosine_sim is None:
        print("Model Content-Based belum siap.")
        return

    # Membuat mapping antara judul film dan indeks DataFrame
    indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()

    try:
        idx = indices[movie_title]
    except KeyError:
        print(f"Film dengan judul '{movie_title}' tidak ditemukan.")
        return

    # Mendapatkan skor kesamaan film yang diberikan dengan semua film lain
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Mengurutkan film berdasarkan skor kesamaan
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Mendapatkan skor dari N film paling mirip (tidak termasuk film itu sendiri)
    sim_scores = sim_scores[1:top_n+1]

    # Mendapatkan indeks film
    movie_indices = [i for i in sim_scores]

    # Mengembalikan N film paling mirip
    recommended_movies = movies_df['title'].iloc[movie_indices].tolist()
    print(f"\nRekomendasi Content-Based untuk '{movie_title}': {recommended_movies}")
    return recommended_movies

# --- 3. Collaborative Filtering ---
def train_collaborative_filtering_model(ratings_df):
    """Melatih model collaborative filtering menggunakan SVD dari Surprise."""
    if ratings_df is None:
        return None
        
    print("\nMelatih model Collaborative Filtering...")
    # Reader diperlukan untuk mem-parse file atau DataFrame
    reader = Reader(rating_scale=(ratings_df['rating'].min(), ratings_df['rating'].max())) # [5, 6, 7, 8, 9, 10]

    # Memuat data dari pandas DataFrame
    # Kolom harus dalam urutan: user, item, rating
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader) # [5, 6, 7, 8, 9, 10]

    # Membangun trainset dari seluruh data
    trainset = data.build_full_trainset() # [5, 7, 9, 11]

    # Menggunakan algoritma SVD (Singular Value Decomposition)
    algo_svd = SVD() # [12, 5, 6, 3, 7, 8, 9, 10, 13, 11]
    algo_svd.fit(trainset) # [5, 6, 7, 8, 9, 10, 13, 11]
    print("Model Collaborative Filtering (SVD) selesai dilatih.")

    # Menyimpan model SVD
    try:
        with open('models/collaborative_model.pkl', 'wb') as f_collab:
            pickle.dump(algo_svd, f_collab) # [5, 11]
        print("Model Collaborative Filtering (SVD) berhasil disimpan.")
    except Exception as e:
        print(f"Gagal menyimpan model Collaborative Filtering: {e}")
        
    return algo_svd

def get_collaborative_filtering_recommendations(user_id, algo_svd, movies_df, ratings_df, top_n=10):
    """Mendapatkan rekomendasi film untuk pengguna berdasarkan collaborative filtering."""
    if algo_svd is None or movies_df is None or ratings_df is None:
        print("Model Collaborative Filtering belum siap.")
        return

    # Dapatkan daftar semua movieId unik
    all_movie_ids = movies_df['movieId'].unique()

    # Dapatkan daftar movieId yang sudah dirating oleh pengguna
    rated_movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId'].unique()

    # Prediksi rating untuk film yang belum dirating pengguna
    predictions =
    for movie_id in all_movie_ids:
        if movie_id not in rated_movie_ids:
            # predict() mengembalikan objek Prediction
            #.est adalah estimasi rating
            predicted_rating = algo_svd.predict(uid=user_id, iid=movie_id).est # [5, 6, 8, 9, 14, 11]
            predictions.append((movie_id, predicted_rating))

    # Urutkan prediksi berdasarkan estimasi rating tertinggi
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Ambil top_n film
    top_predictions = predictions[:top_n]

    # Dapatkan judul film dari movieId
    recommended_movie_ids = [pred for pred in top_predictions]
    recommended_movies = movies_df[movies_df['movieId'].isin(recommended_movie_ids)]['title'].tolist()
    
    print(f"\nRekomendasi Collaborative Filtering untuk User ID {user_id}: {recommended_movies}")
    return recommended_movies

# --- 4. Hybrid Recommendation (Contoh Sederhana) ---
def get_hybrid_recommendations(user_id, liked_movie_title, movies_df, ratings_df, algo_svd, cosine_sim, top_n=10):
    """Menggabungkan rekomendasi dari content-based dan collaborative filtering."""
    print(f"\nMenghasilkan rekomendasi Hybrid untuk User ID {user_id} berdasarkan film '{liked_movie_title}'...")
    
    content_recs = get_content_based_recommendations(liked_movie_title, movies_df, cosine_sim, top_n=top_n)
    collab_recs = get_collaborative_filtering_recommendations(user_id, algo_svd, movies_df, ratings_df, top_n=top_n)

    # Gabungkan hasil dan hilangkan duplikat, pertahankan urutan dari collaborative filtering jika ada tumpang tindih
    hybrid_recs_set = set(collab_recs)
    hybrid_recs_list = list(collab_recs) # Mulai dengan rekomendasi kolaboratif

    for movie in content_recs:
        if movie not in hybrid_recs_set:
            hybrid_recs_list.append(movie)
            hybrid_recs_set.add(movie)
            
    final_recommendations = hybrid_recs_list[:top_n] # [3]
    print(f"\nRekomendasi Hybrid (gabungan): {final_recommendations}")
    return final_recommendations

# --- Main Execution ---
if __name__ == "__main__":
    # Tentukan path ke file CSV Anda
    # Pastikan direktori 'models/' sudah ada untuk menyimpan file.pkl
    # Contoh:./data/movies.csv atau C:/Users/Anda/Documents/data/movies.csv
    MOVIES_CSV_PATH = 'data/movies.csv'  # Ganti dengan path file movies.csv Anda
    RATINGS_CSV_PATH = 'data/ratings.csv' # Ganti dengan path file ratings.csv Anda

    # Buat direktori 'models' jika belum ada
    import os
    if not os.path.exists('models'):
        os.makedirs('models')

    # Muat data
    movies_df, ratings_df = load_data(MOVIES_CSV_PATH, RATINGS_CSV_PATH)

    if movies_df is not None and ratings_df is not None:
        # Latih model Content-Based
        tfidf_vectorizer, cosine_sim_matrix = train_content_based_model(movies_df)

        # Latih model Collaborative Filtering
        svd_model = train_collaborative_filtering_model(ratings_df)

        # --- Contoh Penggunaan ---
        # Pilih pengguna dan film untuk pengujian
        example_user_id = 1  # Ganti dengan user ID yang ada di data Anda
        example_movie_title = 'Toy Story (1995)' # Ganti dengan judul film yang ada di data Anda

        # Dapatkan rekomendasi Content-Based
        if tfidf_vectorizer and cosine_sim_matrix: # Periksa apakah model berhasil dilatih
             get_content_based_recommendations(example_movie_title, movies_df, cosine_sim_matrix)

        # Dapatkan rekomendasi Collaborative Filtering
        if svd_model: # Periksa apakah model berhasil dilatih
            get_collaborative_filtering_recommendations(example_user_id, svd_model, movies_df, ratings_df)

        # Dapatkan rekomendasi Hybrid
        if svd_model and cosine_sim_matrix: # Periksa apakah kedua model berhasil dilatih
            get_hybrid_recommendations(example_user_id, example_movie_title, movies_df, ratings_df, svd_model, cosine_sim_matrix)
        
        print("\n--- Selesai ---")
        print("Model-model (jika berhasil dilatih) telah disimpan di direktori 'models/'.")
        print("Anda dapat memuat model-model ini di aplikasi Flask Anda.")
        print("Contoh memuat model SVD: with open('models/collaborative_model.pkl', 'rb') as f: loaded_svd_model = pickle.load(f)")
        print("Contoh memuat TF-IDF Vectorizer: with open('models/content_vectorizer.pkl', 'rb') as f: loaded_tfidf = pickle.load(f)")
        print("Contoh memuat Cosine Similarity Matrix: with open('models/content_similarity_matrix.pkl', 'rb') as f: loaded_cosine_sim = pickle.load(f)")

    else:
        print("Tidak dapat melanjutkan karena gagal memuat data.")