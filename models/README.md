# Models Directory

Place trained retrieval artifacts here before running the app.

Expected structure:

```text
models/
├── tv1/
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_matrix.pkl
│   ├── bm25.pkl
│   └── corpus_tokens.pkl
├── tv2/
│   ├── embeddings.npy
│   ├── faiss_flat.index
│   └── config.pkl
└── tv3/
    ├── tv3_tfidf_vec.pkl
    ├── tv3_tfidf_mat.pkl
    └── tv3_bm25.pkl
```

These files are intentionally not tracked in git because they are large binary artifacts.
