# ArXIRch - ArXiv Information Retrieval System

ArXIRch là hệ thống truy xuất thông tin cho tập paper arXiv CS. Project kết hợp các phương pháp keyword retrieval, semantic retrieval và document-to-document recommendation để tìm paper liên quan theo truy vấn của người dùng.

Ứng dụng được xây dựng bằng Flask, có giao diện web để nhập truy vấn, chọn phương pháp tìm kiếm, lọc theo category/năm và hiển thị danh sách paper được xếp hạng.

Datasets: https://www.kaggle.com/datasets/Cornell-University/arxiv

## Chức Năng Chính

- Tìm kiếm paper theo từ khóa bằng TF-IDF và BM25.
- Tìm kiếm Boolean với các chế độ AND, OR, NOT.
- Tìm kiếm ngữ nghĩa bằng SPECTER embedding và FAISS.
- Gợi ý paper liên quan từ một paper mẫu bằng D2D TF-IDF và D2D BM25.
- So sánh nhiều phương pháp trong chế độ Compare All.
- Lọc kết quả theo category, năm xuất bản và số lượng top K.
- Hiển thị score, rank, title, category, năm, arXiv id và abstract rút gọn.

## Cấu Trúc Project

```text
.
├── app.py                         # Flask app và API routes
├── search.py                      # Load data/model và các hàm tìm kiếm
├── requirements.txt               # Python dependencies
├── data/
│   ├── arxiv_preprocessed.csv     # Dataset đã tiền xử lý
│   └── arxiv-metadata-oai-snapshot.json
├── models/
│   ├── tv1/                       # TF-IDF và BM25 keyword models
│   ├── tv2/                       # SPECTER embeddings và FAISS index
│   └── tv3/                       # D2D recommendation models
└── templates/
│   └── index.html                 # Giao diện web
├── notebook/
    └── tv1.ipynb
    └── tv2.ipynb
    └── tv3.ipynb
```

## Các Phương Pháp Tìm Kiếm

### TV1 - Keyword Retrieval

**TF-IDF + Cosine Similarity**  
Biến query và paper thành vector TF-IDF, sau đó tính cosine similarity để xếp hạng. Phương pháp này dễ giải thích, chạy ổn định và phù hợp với query có từ khóa rõ ràng.

**BM25**  
Tính điểm dựa trên tần suất từ, độ dài document và độ hiếm của từ khóa. BM25 thường là baseline mạnh cho bài toán search vì cân bằng tốt hơn TF-IDF trong nhiều trường hợp.

**Boolean Search**  
Hỗ trợ các mode AND, OR, NOT để lọc paper theo điều kiện từ khóa rõ ràng.

### TV2 - Semantic Retrieval

Semantic retrieval sử dụng model `allenai-specter` để encode title/abstract thành vector ngữ nghĩa. FAISS được dùng để tìm nearest neighbors nhanh trên tập 100,000 paper.

Phương pháp này phù hợp với query tự nhiên, query có từ đồng nghĩa hoặc khi người dùng mô tả ý tưởng thay vì nhập đúng từ khóa trong paper.

### TV3 - Document-to-Document Recommendation

D2D recommendation nhận đầu vào là title paper hoặc query gần với một paper seed. Từ paper seed, hệ thống tìm các paper liên quan bằng:

- D2D TF-IDF
- D2D BM25

Phương pháp này phù hợp khi người dùng đã có một paper mẫu và muốn tìm các paper cùng chủ đề.

## Cách Hiểu Điểm Score Trên Giao Diện

Score hiển thị trên giao diện là **điểm truy xuất của từng paper**, không phải chỉ số evaluation như P@K, NDCG@K hay F1@K.

- **TF-IDF**: cosine similarity giữa query và paper, thường nằm trong khoảng `0..1`, giao diện hiển thị dạng phần trăm.
- **Semantic SPECTER**: inner product/cosine similarity giữa embedding query và embedding paper, thường hiển thị dạng phần trăm.
- **D2D TF-IDF**: độ giống nhau giữa paper seed và paper kết quả, thường hiển thị dạng phần trăm.
- **BM25**: điểm BM25 thô, có thể lớn hơn 1, giao diện hiển thị dạng số thập phân.
- **D2D BM25**: điểm BM25 thô giữa paper seed và paper khác.
- **Boolean**: số lượng token trong query match với paper.

Lưu ý: score giữa các phương pháp không nên so sánh trực tiếp với nhau, vì mỗi method có thang điểm khác nhau. Score chủ yếu dùng để xếp hạng kết quả trong cùng một phương pháp.

## Cài Đặt Và Chạy Ứng Dụng

### 1. Tạo môi trường ảo

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Cài dependencies

```bash
pip install -r requirements.txt
```

### 3. Kiểm tra data và models

Cần đảm bảo các file sau tồn tại:

```text
data/arxiv_preprocessed.csv
models/tv1/tfidf_vectorizer.pkl
models/tv1/tfidf_matrix.pkl
models/tv1/bm25.pkl
models/tv1/corpus_tokens.pkl
models/tv2/embeddings.npy
models/tv2/faiss_flat.index
models/tv2/config.pkl
models/tv3/tv3_tfidf_vec.pkl
models/tv3/tv3_tfidf_mat.pkl
models/tv3/tv3_bm25.pkl
```

### 4. Chạy Flask app

```bash
python app.py 
python3 app.py
```

Mặc định ứng dụng chạy tại:

```text
http://localhost:5001
```

## API

Endpoint tìm kiếm:

```text
GET /api/search
```

Tham số:

- `q`: query người dùng nhập.
- `method`: `tfidf`, `bm25`, `boolean`, `semantic`, `d2d_tfidf`, `d2d_bm25`, `all`.
- `k`: số kết quả cần trả về.
- `cat`: category filter, ví dụ `cs.LG`, `cs.CV`, `cs.IR`.
- `year`: năm xuất bản.
- `mode`: chế độ Boolean, gồm `AND`, `OR`, `NOT`.

Ví dụ:

```text
/api/search?q=transformer attention natural language processing&method=semantic&k=10
```

Endpoint thống kê:

```text
GET /api/stats
```

## Kết Quả Evaluation Tóm Tắt

Evaluation chung được thực hiện trên 100,000 paper và 11 query test. Các chỉ số chính gồm Precision@K, Recall@K, NDCG@K, F1@K và latency.

Tại K=10:

| Method | P@10 | NDCG@10 | Latency |
|---|---:|---:|---:|
| Semantic | 0.9273 | 0.9867 | 94.3 ms |
| TF-IDF | 0.9273 | 0.9795 | 387.8 ms |
| D2D-BM25 | 0.8909 | 0.9599 | 3222.2 ms |
| D2D-TFIDF | 0.8818 | 0.9593 | 547.1 ms |
| BM25 | 0.9000 | 0.9589 | 214.0 ms |
| Random | 0.5273 | 0.7222 | 2.3 ms |

Semantic là phương pháp tốt nhất trong evaluation chung theo NDCG@10, đồng thời có latency thấp hơn TF-IDF và các phương pháp D2D. BM25 vẫn là baseline mạnh, dễ giải thích và phù hợp với keyword search. D2D-BM25 phù hợp hơn cho bài toán gợi ý paper liên quan từ paper mẫu.

## Hạn Chế

- Ground truth evaluation dựa trên category overlap, nên chỉ phản ánh mức độ liên quan theo lĩnh vực, chưa chắc đánh giá đúng mức độ tương đồng nội dung chi tiết.
- Các category lớn như `cs.LG`, `cs.CV`, `stat.ML` có thể làm điểm P@K/NDCG@K cao hơn thực tế.
- Recall và F1 thấp vì tổng số paper relevant trong corpus rất lớn.
- Score hiển thị trên UI khác thang đo giữa các method, nên không nên so sánh score trực tiếp giữa TF-IDF, BM25 và Semantic.
- Compare All hiện đang so sánh TF-IDF, BM25, Semantic và D2D-TFIDF; D2D-BM25 chưa được đưa vào cột compare.

## Tác Giả / Phân Công

- TV1: Keyword Retrieval - TF-IDF, BM25, Boolean Search.
- TV2: Semantic Retrieval - SPECTER embedding và FAISS.
- TV3: Document-to-Document Recommendation - D2D TF-IDF và D2D BM25.
- Evaluation chung: so sánh các phương pháp bằng P@K, R@K, NDCG@K, F1@K và latency.
