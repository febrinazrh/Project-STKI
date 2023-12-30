import pickle
import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

warnings.filterwarnings("ignore")

# Membaca data set
books = pd.read_csv("books.csv")

# Inisialisasi vectorizer
#mengubah judul buku menjadi vektor tf idf
vectorizer = TfidfVectorizer()
X_new = vectorizer.fit_transform([x.lower() for x in books['title']])

#pencarian dengan contoh query blood
query = 'blood'
query_vec = vectorizer.transform([query])
#dengan menghitung kemiripan kosinus antara query dan judul buku dalam data set
similarity = cosine_similarity(query_vec, X_new).flatten()

#penyortingan dari hasil pencarian dari kemiripan kosinus seacra mennurun
test = np.argsort(-similarity) # sorting descending and get the index
result = books.iloc[test]
result['title'].head()

match_idx = np.where(similarity != 0)[0]
indices = np.argsort(-similarity[match_idx])
correct_indices = match_idx[indices]
result = books.iloc[correct_indices]

result['title'].head()

# Konversi kolom 'average_rating' dan 'ratings_count' menjadi tipe data numerik
books['average_rating'] = pd.to_numeric(books['average_rating'], errors='coerce')
books['ratings_count'] = pd.to_numeric(books['ratings_count'], errors='coerce')

# Cek apakah terdapat nilai-nilai non-numerik (NaN) setelah konversi
if books['average_rating'].isna().any() or books['ratings_count'].isna().any():
    print("Ada nilai non-numerik dalam kolom 'average_rating' atau 'ratings_count'")
else:
    # Hitung 'score' dengan np.log
    books['score'] = np.log(books['average_rating'] * books['ratings_count'])


books['score'] = np.log(books['average_rating'] *  books['ratings_count'])


# Fungsi search engine
def search_engine(word, limit=5):
    word = re.sub('[^a-zA-Z0-9 ]', '', word.lower()) #pembersihan kata kunci
    query_vec = vectorizer.transform([word]) #vektorisasi kata mnjdi nilai tf idf utk mengukur kemiripan
    similarity = cosine_similarity(query_vec, X_new).flatten() #menghitung kemiripan kosinus dari kata kunci pencarian dengan judul pada dataset

    filtered = np.where(similarity != 0)[0] #menghilangkn judul yang tidak memiliki kemiripan sama sekali]
    indices = np.argsort(-similarity[filtered]) #mengurutkan secara descending
    correct_indices = filtered[indices] #mengambil indeks yang sudah diurutkan
    result = books.iloc[correct_indices]

    if not len(result): #memeriksa hasil pencarian kosong atau tidak
        return 'result not found'
    #hitung semua judul dengan mengalikan skor kemiripan dengan skor keseluruhan yang dihitung sebelumnya
    overall = result['score'] * similarity[correct_indices]
    #mengembalikan hasil pencarian
    return result.loc[overall.sort_values(ascending=False).index].head(limit)

# Judul aplikasi
st.title('🔍 Search Engine Film')
st.write('Gunakan search engine ini untuk menemukan informasi dengan cepat!')

# Input pencarian dengan efek animasi
search_term = st.text_input('Masukkan kata kunci pencarian:', value='', key='search_input')

# Menampilkan hasil pencarian
if st.button('Cari'):
    result = search_engine(search_term)
    st.subheader('Hasil Pencarian:')
    if isinstance(result, pd.DataFrame):
        st.write(result[['title', 'authors', 'average_rating', 'ratings_count']].head())
    else:
        st.warning('Pencarian tidak ditemukan.')

# Sidebar dengan opsi penggunaan aplikasi
st.sidebar.header('Pilihan Aplikasi')
option = st.sidebar.radio('Pilih Tipe Pencarian:', ['title', 'authors', 'rating'])

# Menampilkan data dalam bentuk kartu dengan animasi
st.sidebar.subheader('Info Dataset:')
st.sidebar.write(f"Jumlah Baris: {books.shape[0]}")
st.sidebar.write(f"Kolom: {', '.join(books.columns)}")

# Menampilkan beberapa data teratas dari dataset
st.sidebar.subheader('Data Awal:')
st.sidebar.dataframe(books.head())
