# Wikipedia Search Engine 🔎
### Information Retrieval Final Project (2025-2026)

**Authors:**
* **Hadar Knafo** (ID: 206747792)
* **Lital Kupchick** (ID: 318567914)

---

## Overview
A large-scale search engine for the entire English Wikipedia dump (>6 million articles), built with **PySpark** for distributed indexing and **Flask** for serving search queries. The system is deployed on **Google Cloud Platform (GCP)** and utilizes a hybrid ranking strategy combining TF-IDF, Title matching, and Anchor text signals.

**Live Deployment IP:** `http://35.193.251.255:8080`

## Architecture

The search engine is built on several key components:

### 1. Inverted Indices (Distributed Storage)
* **Body Index:** Full-text index using TF-IDF scoring.
* **Title Index:** Optimized index for article titles (high precision).
* **Anchor Index:** Index based on anchor text from incoming hyperlinks.
* *Storage:* All indices are stored in a **Google Cloud Storage (GCS)** bucket (`gs://ir-wiki-hadar`) as split binary files to allow efficient random access and low-latency retrieval.

### 2. Ranking Strategy
The search engine uses a **weighted hybrid approach**. Through experimentation, we found that article titles are the strongest signal for relevance in Wikipedia.

**Final Weights Configuration:**
* **Title Weight:** 0.85 (Focus on high-precision matches)
* **Body Weight:** 0.10 (Support for content-based recall)
* **Anchor Weight:** 0.05 (Additional collaborative signal)

### 3. Performance & Results
* **Lazy Loading:** The system loads only lightweight term dictionaries into RAM at startup. Heavy posting lists are read from GCS only upon request to optimize memory usage.
* **Binary Packing:** Posting lists are compressed into 6-byte blocks (4 bytes DocID + 2 bytes TF).
* **Evaluation Results:**
    * **MAP@40:** 0.0923 (Hybrid model)
    * **Avg Latency:** ~3.2 seconds per query

## Project Structure

```
.
├── search_frontend.py              # Flask API server with search endpoints
├── inverted_index_gcp.py          # Inverted index data structure and I/O operations
├── Part1_Build_Body_and_Title_Index.ipynb.ipynb  # PySpark notebook for body/title indexing
├── Part2_Build_Anchor_and_PageViews.ipynb        # PySpark notebook for anchor/PageRank
├── queries_train.json             # Training queries for evaluation
├── run_frontend_in_gcp.sh         # Shell script to deploy on GCP Compute Engine
├── run_frontend_in_colab.ipynb    # Notebook for testing frontend in Colab
├── startup_script_gcp.sh          # GCP VM startup script
└── README.md                      # This file
```

### Key Files

#### `search_frontend.py`
Main Flask application providing the following endpoints:
- `/search` - Hybrid search (combines title and body with weighted scoring)
- `/search_body` - Body-only search using TF-IDF cosine similarity
- `/search_title` - Title-only search with term frequency counting
- `/search_anchor` - Anchor text search
- `/get_pagerank` - Retrieve PageRank scores for document IDs
- `/get_pageview` - Retrieve page view counts for document IDs

#### `inverted_index_gcp.py`
Core inverted index implementation with:
- Binary encoding of posting lists (6 bytes per entry: 4 bytes doc_id + 2 bytes term frequency)
- Multi-file reader/writer for handling large-scale data
- Efficient streaming from GCS buckets
- Support for both local and cloud storage

#### Index Building Notebooks
- **Part1**: Builds body and title inverted indices using PySpark on Wikipedia XML dump
- **Part2**: Builds anchor text index and computes PageRank using iterative graph algorithms

## Search Strategies

### Hybrid Search (Default)
Combines multiple signals with weighted scoring:
- **Title Match**: 0.7 weight (high precision signal)
- **Body Match**: 0.3 × TF-IDF cosine similarity
- Returns top 100 results by relevance

### Body Search
Pure content-based ranking using TF-IDF:
- Computes: `score = Σ (tf × idf) / doc_norm` for each query term
- Normalized by document length for fair comparison
- Returns top 100 results

### Title Search
Simple but effective for navigational queries:
- Counts matching query terms in article titles
- No normalization (titles are short)
- Good for finding specific articles

### Anchor Search
Uses collaborative filtering signal:
- Searches text of hyperlinks pointing to articles
- Captures how Wikipedia editors describe articles
- Useful for alternative names and descriptions

## How to Run

### Prerequisites
- Python 3.7+
- Google Cloud SDK configured with project access
- Flask and required dependencies

### Installation

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd ir_proj_20251213
```

2. **Install dependencies**:
```bash
pip install flask google-cloud-storage
```

3. **Set up GCP credentials**:
```bash
# Set your GCP project ID
export GOOGLE_CLOUD_PROJECT=your-project-id

# Authenticate (if not already done)
gcloud auth application-default login
```

4. **Update configuration**:
   - Edit `search_frontend.py` and set `BUCKET_NAME` to your GCS bucket
   - Edit `inverted_index_gcp.py` and set `PROJECT_ID` to your GCP project

### Running Locally

```bash
# Run the Flask server
python search_frontend.py

# The API will be available at http://localhost:8080
```

### Running on GCP Compute Engine

1. **Deploy using the provided script**:
```bash
bash run_frontend_in_gcp.sh
```

2. **Or manually**:
```bash
# Create a VM instance
gcloud compute instances create wikipedia-search \
    --zone=us-central1-a \
    --machine-type=n1-standard-2 \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=30GB

# Copy files to the instance
gcloud compute scp search_frontend.py inverted_index_gcp.py wikipedia-search:~

# SSH and run
gcloud compute ssh wikipedia-search
python3 search_frontend.py
```

### Testing the API

```bash
# Search with hybrid strategy
curl "http://localhost:8080/search?query=artificial+intelligence"

# Search in body text only
curl "http://localhost:8080/search_body?query=machine+learning"

# Search in titles
curl "http://localhost:8080/search_title?query=python+programming"

# Get PageRank scores
curl -X POST http://localhost:8080/get_pagerank \
  -H "Content-Type: application/json" \
  -d '[12345, 67890]'

# Get page view counts
curl -X POST http://localhost:8080/get_pageview \
  -H "Content-Type: application/json" \
  -d '[12345, 67890]'
```

## Performance Considerations

- **Index Size**: Body index ~20GB, Title index ~500MB, Anchor index ~2GB
- **Query Latency**: Typically 100-500ms depending on query complexity
- **Scaling**: Can handle 100+ concurrent queries with proper GCP instance sizing
- **Caching**: Frequently accessed posting lists are cached in memory

## Technical Details

### Binary Encoding
Posting lists use compact binary encoding:
- Each (doc_id, term_frequency) pair: 6 bytes
- doc_id: 4 bytes (big-endian integer)
- term_frequency: 2 bytes (big-endian integer)
- Formula: `(doc_id << 16 | tf) encoded as 6-byte integer`

### TF-IDF Calculation
```python
idf = log10(N / df)  # N = total docs, df = document frequency
score = (tf × idf) / doc_norm  # Normalized by document length
```

### Hybrid Ranking
```python
final_score = 0.7 × title_matches + 0.3 × body_tfidf_score
```

## Data Sources

- **Wikipedia Dump**: English Wikipedia XML dump (latest version)
- **PageRank**: Computed from Wikipedia's internal link graph
- **PageViews**: Historical page view statistics from Wikimedia

## Future Improvements

- [ ] Add query expansion with synonyms
- [ ] Implement BM25 ranking as an alternative to TF-IDF
- [ ] Add support for phrase queries
- [ ] Integrate machine learning ranking (Learning to Rank)
- [ ] Add caching layer (Redis) for frequent queries
- [ ] Implement query autocompletion
- [ ] Add support for multilingual Wikipedia

## Authors

Information Retrieval Course Project

## License

This project is for educational purposes.

## Acknowledgments

- Wikipedia for providing the open data dump
- Apache Spark for distributed computing framework
- Google Cloud Platform for infrastructure
