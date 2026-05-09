# search.py — Load models + tất cả hàm tìm kiếm
import re, pickle, time
import numpy as np
import pandas as pd
import faiss
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords', quiet=True)

STOP = set(stopwords.words('english'))
PS   = PorterStemmer()

# PATHS — khớp chính xác với output của 3 notebooks
DATA_PATH = 'data/arxiv_preprocessed.csv'
TV1_DIR   = 'models/tv1'
TV2_DIR   = 'models/tv2'
TV3_DIR   = 'models/tv3'

# LOAD DATA + MODELS
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['text_tfidf','text_semantic','text_recommend'])
df = df.reset_index(drop=True)
title_to_idx = pd.Series(df.index, index=df['title'].str.lower().str.strip())
print(f"  {len(df):,} papers loaded")

print("Loading TV1 models (TF-IDF + BM25)...")
tfidf_vec     = pickle.load(open(f'{TV1_DIR}/tfidf_vectorizer.pkl','rb'))
tfidf_mat     = pickle.load(open(f'{TV1_DIR}/tfidf_matrix.pkl','rb'))
bm25          = pickle.load(open(f'{TV1_DIR}/bm25.pkl','rb'))
corpus_tokens = pickle.load(open(f'{TV1_DIR}/corpus_tokens.pkl','rb'))

print("Loading TV2 models (SPECTER + FAISS)...")
sbert      = SentenceTransformer('allenai-specter')
embeddings = np.load(f'{TV2_DIR}/embeddings.npy').astype('float32')
faiss_idx  = faiss.read_index(f'{TV2_DIR}/faiss_flat.index')

print("Loading TV3 models (D2D)...")
tv3_tfidf_vec = pickle.load(open(f'{TV3_DIR}/tv3_tfidf_vec.pkl','rb'))
tv3_tfidf_mat = pickle.load(open(f'{TV3_DIR}/tv3_tfidf_mat.pkl','rb'))
tv3_count_vec = pickle.load(open(f'{TV3_DIR}/tv3_count_vec.pkl','rb'))
tv3_count_mat = pickle.load(open(f'{TV3_DIR}/tv3_count_mat.pkl','rb'))
tv3_bm25      = pickle.load(open(f'{TV3_DIR}/tv3_bm25.pkl','rb'))
tv3_tokens    = [str(t).split() for t in df['text_recommend']]

print("All models loaded!")

# HELPERS
def _preprocess(text: str) -> str:
    text = re.sub(r'[^a-z\s]', ' ', text.lower())
    return ' '.join([PS.stem(t) for t in text.split()
                     if t not in STOP and len(t) > 2])

def _fmt(indices, scores=None, method='') -> list:
    """Chuyển list index → list dict để trả về JSON"""
    results = []
    for rank, idx in enumerate(indices):
        if idx < 0 or idx >= len(df): continue
        row = df.iloc[idx]
        results.append({
            'rank':       rank + 1,
            'id':         str(row['id']),
            'title':      str(row['title']),
            'abstract':   str(row['abstract'])[:250] + '...',
            'categories': str(row['categories']),
            'year':       str(row['year']),
            'score':      round(float(scores[rank]), 4) if scores else None,
            'method':     method,
        })
    return results

def _find_idx(paper_title: str):
    """Tìm index của paper theo title — khớp với TV3"""
    import difflib
    title   = paper_title.lower().strip()
    titles  = df['title'].fillna('').str.lower().str.strip()
    exact   = df.index[titles == title]
    if len(exact) > 0: return exact[0]
    contains = df.index[titles.str.contains(title, regex=False)]
    if len(contains) > 0: return contains[0]
    candidates = difflib.get_close_matches(title, titles.tolist(),
                                           n=1, cutoff=0.55)
    if candidates:
        return df.index[titles == candidates[0]][0]
    return None

# ════════════════════════════════════════════════════════════
# TV1: KEYWORD SEARCH
# ════════════════════════════════════════════════════════════
def search_tfidf(query: str, top_k=10, cat='', year='') -> dict:
    t0    = time.time()
    q_vec = tfidf_vec.transform([_preprocess(query)])
    sc    = cosine_similarity(q_vec, tfidf_mat).flatten()
    idx   = sc.argsort()[::-1][:top_k*3]

    # Filter
    rows = df.iloc[idx].copy()
    rows['_score'] = sc[idx]
    if cat:  rows = rows[rows['categories'].str.contains(cat,  na=False)]
    if year: rows = rows[rows['year'].astype(str) == str(year)]
    rows = rows.head(top_k)

    return {
        'results':  _fmt(rows.index.tolist(),
                         rows['_score'].tolist(), 'TF-IDF'),
        'method':   'TF-IDF',
        'time_ms':  round((time.time()-t0)*1000, 1),
    }

def search_bm25(query: str, top_k=10, cat='', year='') -> dict:
    t0     = time.time()
    tokens = _preprocess(query).split()
    sc     = np.array(bm25.get_scores(tokens))
    idx    = sc.argsort()[::-1][:top_k*3]

    rows = df.iloc[idx].copy()
    rows['_score'] = sc[idx]
    if cat:  rows = rows[rows['categories'].str.contains(cat,  na=False)]
    if year: rows = rows[rows['year'].astype(str) == str(year)]
    rows = rows.head(top_k)

    return {
        'results':  _fmt(rows.index.tolist(),
                         rows['_score'].tolist(), 'BM25'),
        'method':   'BM25',
        'time_ms':  round((time.time()-t0)*1000, 1),
    }

def search_boolean(query: str, mode='AND', top_k=10,
                   cat='', year='') -> dict:
    t0     = time.time()
    tokens = _preprocess(query).split()
    if not tokens:
        return {'results': [], 'method': f'Boolean-{mode}', 'time_ms': 0}

    token_sets = df['text_tfidf'].fillna('').str.split().apply(set)

    if mode == 'AND':
        mask = token_sets.apply(lambda w: all(t in w for t in tokens))
    elif mode == 'OR':
        mask = token_sets.apply(lambda w: any(t in w for t in tokens))
    elif mode == 'NOT':
        if len(tokens) >= 2:
            mask = token_sets.apply(lambda w: tokens[0] in w
                                    and tokens[1] not in w)
        else:
            mask = token_sets.apply(lambda w: tokens[0] in w)
    else:
        mask = token_sets.apply(lambda w: any(t in w for t in tokens))

    rows = df[mask].copy()
    rows['match_score'] = rows['text_tfidf'].fillna('').str.split().apply(
        lambda w: sum(t in set(w) for t in tokens)
    )
    if cat:  rows = rows[rows['categories'].str.contains(cat,  na=False)]
    if year: rows = rows[rows['year'].astype(str) == str(year)]
    rows = rows.sort_values(['match_score','year'],
                            ascending=[False,False]).head(top_k)

    return {
        'results':  _fmt(rows.index.tolist(),
                         rows['match_score'].tolist(),
                         f'Boolean-{mode}'),
        'method':   f'Boolean-{mode}',
        'time_ms':  round((time.time()-t0)*1000, 1),
    }

# TV2: SEMANTIC SEARCH
def search_semantic(query: str, top_k=10, cat='', year='') -> dict:
    t0    = time.time()
    q_clean = re.sub(r'\$.*?\$', ' ', query)
    q_emb = sbert.encode(
        [q_clean], normalize_embeddings=True, convert_to_numpy=True
    ).astype('float32')
    sc, indices = faiss_idx.search(q_emb, top_k * 3)

    rows = df.iloc[indices[0]].copy()
    rows['_score'] = sc[0]
    if cat:  rows = rows[rows['categories'].str.contains(cat,  na=False)]
    if year: rows = rows[rows['year'].astype(str) == str(year)]
    rows = rows.head(top_k)

    return {
        'results':  _fmt(rows.index.tolist(),
                         rows['_score'].tolist(), 'Semantic-SPECTER'),
        'method':   'Semantic-SPECTER',
        'time_ms':  round((time.time()-t0)*1000, 1),
    }

# TV3: DOCUMENT-TO-DOCUMENT RETRIEVAL
def search_d2d_tfidf(query: str, top_k=10, cat='', year='') -> dict:
    t0 = time.time()
    idx = _find_idx(query)
    if idx is None:
        # Fallback: tìm paper gần nhất làm seed
        q_vec    = tv3_tfidf_vec.transform([query])
        seed_sc  = cosine_similarity(q_vec, tv3_tfidf_mat).flatten()
        idx      = int(seed_sc.argsort()[-1])

    sc      = linear_kernel(tv3_tfidf_mat[idx], tv3_tfidf_mat).flatten()
    sc[idx] = 0
    top_idx = sc.argsort()[::-1][:top_k*3]

    rows = df.iloc[top_idx].copy()
    rows['_score'] = sc[top_idx]
    if cat:  rows = rows[rows['categories'].str.contains(cat,  na=False)]
    if year: rows = rows[rows['year'].astype(str) == str(year)]
    rows = rows.head(top_k)

    return {
        'results':      _fmt(rows.index.tolist(),
                             rows['_score'].tolist(), 'D2D-TF-IDF'),
        'method':       'D2D-TF-IDF',
        'seed_paper':   str(df.loc[idx, 'title']),
        'time_ms':      round((time.time()-t0)*1000, 1),
    }

def search_d2d_bm25(query: str, top_k=10, cat='', year='') -> dict:
    t0 = time.time()
    idx = _find_idx(query)
    if idx is None:
        q_vec = tv3_tfidf_vec.transform([query])
        idx   = int(cosine_similarity(q_vec, tv3_tfidf_mat)
                    .flatten().argsort()[-1])

    sc      = np.array(tv3_bm25.get_scores(tv3_tokens[idx]))
    sc[idx] = 0
    top_idx = sc.argsort()[::-1][:top_k*3]

    rows = df.iloc[top_idx].copy()
    rows['_score'] = sc[top_idx]
    if cat:  rows = rows[rows['categories'].str.contains(cat,  na=False)]
    if year: rows = rows[rows['year'].astype(str) == str(year)]
    rows = rows.head(top_k)

    return {
        'results':      _fmt(rows.index.tolist(),
                             rows['_score'].tolist(), 'D2D-BM25'),
        'method':       'D2D-BM25',
        'seed_paper':   str(df.loc[idx, 'title']),
        'time_ms':      round((time.time()-t0)*1000, 1),
    }

# SO SÁNH TẤT CẢ
def search_all(query: str, top_k=5, cat='', year='') -> dict:
    return {
        'tfidf':     search_tfidf(query,     top_k, cat, year),
        'bm25':      search_bm25(query,      top_k, cat, year),
        'semantic':  search_semantic(query,  top_k, cat, year),
        'd2d_tfidf': search_d2d_tfidf(query, top_k, cat, year),
    }

# STATS
def get_stats() -> dict:
    return {
        'total_papers': len(df),
        'year_min':     int(df['year'].min()),
        'year_max':     int(df['year'].max()),
        'categories':   df['categories'].str.split().explode()
                          .value_counts().head(8).to_dict(),
    }