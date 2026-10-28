<div align="center">
  <h1>Text Similarity & Content-Based Recommender System</h1>
  <p><strong>Chapter 7 — Natural Language Processing & Text Processing Pipeline</strong></p>

  <p align="center">
    <img src="https://img.shields.io/badge/Language-Python_3-3776AB?style=flat-square&logo=python&logoColor=white" alt="Language">
    <img src="https://img.shields.io/badge/Environment-Jupyter_Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white" alt="Environment">
    <img src="https://img.shields.io/badge/Dataset-TMDB_Movies-01b4e4?style=flat-square" alt="Dataset">
    <img src="https://img.shields.io/badge/Embeddings-Word2Vec_%26_FastText-green?style=flat-square" alt="Embeddings">
    <img src="https://img.shields.io/badge/Status-Complete-success?style=flat-square" alt="Status">
  </p>

  <p align="center">
    Comprehensive natural language processing notebook exploring lexical and semantic similarity metrics, distance functions, content-based movie recommender systems, and free-text search query matching on TMDB movie overviews.
  </p>
</div>

## Tech Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/Seaborn-%233776AB.svg?style=for-the-badge&logo=python&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)

## Project Overview

Repositori ini memuat implementasi mandiri untuk modul **Chapter 7 - Text Similarity** pada mata kuliah *Text Processing* Program Studi Sains Data Universitas Negeri Surabaya. Analisis ini memanfaatkan data hasil *feature engineering* sebelumnya dari korpus *The Movie Database* (TMDB) yang tersimpan di `data/tmdb_preprocessed.csv` serta bobot neural word embeddings (`models/tmdb_word2vec.w2v` dan `models/tmdb_fasttext.fasttext`).

Notebook utama [`Text_Similarity_Rava_Welfee.ipynb`](Text_Similarity_Rava_Welfee.ipynb) menguji secara komparatif pendekatan **Lexical Similarity** (kesamaan kata eksak berbasis representasi vektor jarang) versus **Semantic Similarity** (kesamaan makna kontekstual berbasis representasi continuous dense vector).

## Repository Structure

```text
Text-Similarity/
├── data/
│   └── tmdb_preprocessed.csv                  # Preprocessed TMDB movie metadata and overviews
├── models/
│   ├── tmdb_word2vec.w2v                      # Pre-trained Word2Vec CBOW model
│   └── tmdb_fasttext.fasttext                 # Pre-trained FastText Skip-gram model
├── .gitignore                                 # Git ignore rules
├── requirements.txt                           # Python dependencies
├── Text_Similarity_Rava_Welfee.ipynb          # Main executed text similarity notebook
└── README.md                                  # Project documentation
```

## Theoretical Framework

### 1. Cosine Similarity
Mengukur kosinus sudut antara dua vektor berdimensi tinggi tanpa dipengaruhi oleh panjang magnitudo dokumen:

$$
\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^{V} u_i v_i}{\sqrt{\sum_{i=1}^{V} u_i^2} \sqrt{\sum_{i=1}^{V} v_i^2}}
$$

Jarak kosinus (*Cosine Distance*) dihitung sebagai komplemennya:

$$
d_{\text{cosine}}(\mathbf{u}, \mathbf{v}) = 1 - \cos(\mathbf{u}, \mathbf{v})
$$

### 2. Jaccard Similarity
Mengukur rasio irisan token terhadap gabungan token unik (*Intersection over Union*):

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}, \quad d_{\text{jaccard}}(A, B) = 1 - J(A, B)
$$

### 3. Alternative Distance Functions
Mengevaluasi ragam metrik jarak matematis berpasangan:
- **Euclidean Distance**: $d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$
- **Manhattan Distance**: $d(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{n} |x_i - y_i|$
- **Chebyshev Distance**: $d(\mathbf{x}, \mathbf{y}) = \max_i (|x_i - y_i|)$

### 4. Semantic Similarity via Continuous Dense Embeddings
Dokumen direpresentasikan ke dalam vektor rata-rata berdimensi 100 (*mean document vector pooling*):

$$
\mathbf{d} = \frac{1}{|d_{\text{valid}}|} \sum_{w \in d_{\text{valid}}} \mathbf{v}_w
$$

Kedekatan makna antar film dihitung dengan memproyeksikan vektor kontinu tersebut ke dalam metrik kosinus.

## Applied Use Cases

1. **Content-Based Movie Recommender System (Item-to-Item)**:
   - Menghitung matriks pairwise similarity $298 \times 298$.
   - Fungsi rekomendasi `recommend_movies_by_title()` menghasilkan Top-5 film serupa untuk judul target (misalnya *Resident Evil* dan *Spider-Man: Brand New Day*).
2. **Free-Text Search Query Matching (User Query-to-Item)**:
   - Menguji pencarian teks bebas (seperti *"superhero fight crime save city battle"* dan *"dangerous virus outbreak apocalypse dead infection"*).
   - Membandingkan hasil pencarian secara simultan antara **TF-IDF**, **Jaccard**, dan **Word2Vec**.
3. **Analisis Komparatif Leksikal vs Semantik**:
   - Representasi leksikal (TF-IDF) unggul dalam ketepatan kata kunci spesifik (*high keyword precision*).
   - Representasi semantik (Word2Vec) mampu menangkap hubungan naratif dan sinonim meskipun tidak ada kata yang identik (*semantic generalization*).

## Installation and Execution

```bash
# 1. Clone repository
git clone https://github.com/welfeesukardi/Text-Similarity.git
cd Text-Similarity

# 2. Buat virtual environment (opsional)
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
# source venv/bin/activate    # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Jalankan notebook
jupyter notebook Text_Similarity_Rava_Welfee.ipynb
```
Notebook telah dilengkapi *adaptive path handling* untuk menjamin eksekusi bebas kendala di berbagai sistem operasi maupun Google Colab.
