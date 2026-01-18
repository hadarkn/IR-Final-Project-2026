# Search Frontend for Wikipedia Search Engine
# This module implements a Flask-based search API that retrieves and ranks Wikipedia articles
# based on various search strategies (body, title, anchor text) and ranking metrics (PageRank, PageViews)

from flask import Flask, request, jsonify
from collections import Counter
import os, re, pickle, math
from google.cloud import storage
from inverted_index_gcp import InvertedIndex

class SearchFrontend:
    """
    Main search engine class that handles Wikipedia article search and retrieval.
    Implements multiple search strategies: body text search, title search, anchor text search,
    and a hybrid approach combining multiple signals.
    """
    
    def __init__(self):
        # GCP bucket name where all index files and metadata are stored
        self.BUCKET_NAME = 'ir-wiki-hadar'
        
        # Metadata dictionaries - loaded lazily on first use
        self.page_views = None      # Dictionary mapping doc_id -> view count
        self.page_rank = None       # Dictionary mapping doc_id -> PageRank score
        self.id_to_title = None     # Dictionary mapping doc_id -> article title
        self.doc_norms = None       # Dictionary mapping doc_id -> document length norm (for cosine similarity)
        
        # Inverted indices for different text fields - loaded lazily on first use
        self.body_index = None      # Inverted index for article body text
        self.title_index = None     # Inverted index for article titles
        self.anchor_index = None    # Inverted index for anchor text (text of links pointing to articles)
        
        # Google Cloud Storage client (initialized lazily)
        self._storage_client = None

        # Common English stopwords to filter out during tokenization
        self.stopwords = frozenset(['the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'that', 'for', 'you', 'he', 'was', 'on', 'are', 'with', 'as', 'i', 'his', 'they', 'be', 'at', 'one', 'have', 'this', 'from', 'or', 'had', 'by', 'not', 'word', 'but', 'what', 'some', 'we', 'can', 'out', 'other', 'were', 'all', 'there', 'when', 'up', 'use', 'your', 'how', 'said', 'an', 'each', 'she'])
        
        # Regex pattern to extract words (2-24 characters, can include hashtags, mentions, hyphens, apostrophes)
        self.re_word = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

    def _client(self):
        """
        Lazy initialization of Google Cloud Storage client.
        Returns the existing client or creates a new one if not initialized.
        """
        if self._storage_client is None:
            self._storage_client = storage.Client()
        return self._storage_client

    def _download_pickle(self, gcs_path, local_path):
        """
        Download and load a pickle file from Google Cloud Storage.
        First checks if the file exists locally to avoid redundant downloads.
        
        Args:
            gcs_path: Path to file in GCS bucket
            local_path: Local path to save/load the file
        
        Returns:
            The unpickled Python object, or None if download/load fails
        """
        # Try to load from local cache first
        if os.path.exists(local_path):
            try:
                with open(local_path, 'rb') as f:
                    return pickle.load(f)
            except: pass 
        
        # Download from GCS and save locally
        try:
            bucket = self._client().bucket(self.BUCKET_NAME)
            blob = bucket.blob(gcs_path)
            blob.download_to_filename(local_path)
            with open(local_path, 'rb') as f:
                return pickle.load(f)
        except: return None


    def _load_extra_locs(self, index, prefix):
        """
        Load additional posting list locations from split files in GCS.
        The inverted index is split across multiple files for efficiency.
        This method loads all the _posting_locs.pickle files and merges them into the main index.
        
        Args:
            index: The InvertedIndex object to update
            prefix: GCS path prefix to search for posting location files
        """
        if not index: return
        try:
            bucket = self._client().bucket(self.BUCKET_NAME)
            blobs = bucket.list_blobs(prefix=prefix)
            # Find and load all posting location files
            for blob in blobs:
                if '_posting_locs.pickle' in blob.name:
                    with blob.open("rb") as f:
                        locs = pickle.load(f)
                        # Merge posting locations into the main index
                        index.posting_locs.update(locs)
                        # Update document frequency for each term
                        for word in locs.keys():
                            index.df[word] = index.df.get(word, 0) + 1
            
            print(f"DEBUG: Total terms now in index.df: {len(index.df)}")
        except Exception as e:
            print(f"DEBUG: Error in _load_extra_locs: {e}")




    def load_metadata(self):
        """
        Load all metadata dictionaries from GCS (lazy loading).
        This includes: document titles, document norms for TF-IDF, page views, and PageRank scores.
        Only loads once on first call.
        """
        if self.id_to_title is None:
            self.id_to_title = self._download_pickle('id_to_title.pkl', 'id_to_title.pkl') or {}
            self.doc_norms = self._download_pickle('doc_norms.pkl', 'doc_norms.pkl') or {}
            self.page_views = self._download_pickle('page_views.pkl', 'page_views.pkl') or {}
            self.page_rank = self._download_pickle('page_rank.pkl', 'page_rank.pkl') or {}

    def load_body_index(self):
        """
        Load the body text inverted index from GCS (lazy loading).
        The body index contains terms from the main text content of Wikipedia articles.
        Also loads split posting location files.
        """
        if self.body_index is None:
            self.body_index = self._download_pickle('postings_gcp/index.pkl', 'index_body.pkl')
            if self.body_index:
                self.body_index.base_dir = 'postings_gcp'
                # Load additional posting locations from split files
                self._load_extra_locs(self.body_index, 'postings_gcp/')

    def load_title_index(self):
        """
        Load the title inverted index from GCS (lazy loading).
        The title index contains terms from Wikipedia article titles.
        Title matches are generally more relevant than body matches.
        """
        if self.title_index is None:
            self.title_index = self._download_pickle('title_index/index.pkl', 'index_title.pkl')
            if self.title_index:
                self.title_index.base_dir = 'title_index'
                # Load additional posting locations from split files
                self._load_extra_locs(self.title_index, 'title_index/')

    def load_anchor_index(self):
        """
        Load the anchor text inverted index from GCS (lazy loading).
        The anchor index contains terms from the text of hyperlinks pointing to Wikipedia articles.
        Anchor text often provides good descriptions of what an article is about.
        """
        if self.anchor_index is None:
            self.anchor_index = self._download_pickle('anchor_index/index.pkl', 'index_anchor.pkl')
            if self.anchor_index:
                self.anchor_index.base_dir = 'anchor_index'
                # Critical: Load split posting location files
                self._load_extra_locs(self.anchor_index, 'anchor_index/')

    def tokenize(self, text):
        """
        Tokenize query text into individual terms.
        
        Process:
        1. Extract words using regex pattern (2-24 chars, allows hyphens/apostrophes)
        2. Convert to lowercase
        3. Filter out stopwords
        
        Args:
            text: Query string to tokenize
        
        Returns:
            List of tokenized terms
        """
        return [t.group().lower() for t in self.re_word.finditer(text) if t.group().lower() not in self.stopwords]

    def _get_posting(self, index, token):
        """
        Retrieve the posting list for a given token from an inverted index.
        
        Args:
            index: The InvertedIndex object to query
            token: The term to look up
        
        Returns:
            List of tuples (doc_id, term_frequency) for documents containing the token
            Returns empty list if token not found or error occurs
        """
        if not index or token not in index.df: return []
        try:
            return index.read_a_posting_list(index.base_dir, token, self.BUCKET_NAME)
        except: return []

    def search_title(self, query):
        """
        Search for documents by matching query terms in article titles.
        
        Ranking: Simple count of matching query terms in title (no TF-IDF).
        Each matching term adds 1 to the document's score.
        
        Args:
            query: Search query string
        
        Returns:
            List of tuples (doc_id, title) sorted by relevance score (descending)
        """
        self.load_metadata(); self.load_title_index()
        tokens = self.tokenize(query); scores = Counter()
        # Count how many query terms appear in each document's title
        for t in tokens:
            for d, _ in self._get_posting(self.title_index, t):
                scores[int(d)] += 1
        res = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(str(d), self.id_to_title.get(int(d), "Unknown")) for d, _ in res]

    def search_anchor(self, query):
        """
        Search for documents by matching query terms in anchor text.
        Anchor text = the clickable text of hyperlinks pointing to the article.
        
        Ranking: Simple count of matching query terms in anchor text.
        Each matching term adds 1 to the document's score.
        
        Args:
            query: Search query string
        
        Returns:
            List of tuples (doc_id, title) sorted by relevance score (descending)
        """
        self.load_metadata(); self.load_anchor_index()
        print(f"DEBUG: Sample keys in anchor index: {list(self.anchor_index.df.keys())[:10]}")
        tokens = self.tokenize(query); scores = Counter()
        # Count how many query terms appear in anchor text pointing to each document
        for t in tokens:
            for d, _ in self._get_posting(self.anchor_index, t):
                scores[int(d)] += 1
        res = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(str(d), self.id_to_title.get(int(d), "Unknown")) for d, _ in res]

    def search(self, query):
        """
        Hybrid search combining title and body text matching.
        
        Ranking formula:
        - Title match: +0.7 per query term (higher weight because titles are more indicative)
        - Body match: +0.3 * (TF-IDF cosine similarity)
          - TF-IDF = (term frequency * inverse document frequency) / document norm
          - IDF = log10(N / document_frequency)
        
        Args:
            query: Search query string
        
        Returns:
            List of top 100 tuples (doc_id, title) sorted by relevance score (descending)
        """
        self.load_metadata(); self.load_title_index(); self.load_body_index()
        tokens = self.tokenize(query); scores = Counter()
        N = len(self.doc_norms) if self.doc_norms else 6000000  # Total number of documents
        
        for t in tokens:
            # Title signal: 0.7 weight for each title match
            if self.title_index:
                for d, _ in self._get_posting(self.title_index, t):
                    scores[int(d)] += 0.7
            
            # Body signal: 0.3 * TF-IDF cosine similarity
            if self.body_index:
                idf = math.log(N / self.body_index.df[t], 10) if t in self.body_index.df else 0
                for d, tf in self._get_posting(self.body_index, t):
                    scores[int(d)] += 0.3 * (tf * idf) / self.doc_norms.get(int(d), 1)
        
        res = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(str(d), self.id_to_title.get(d, "Unknown")) for d, _ in res[:100]]

    def search_body(self, query):
        """
        Search for documents by matching query terms in article body text.
        
        Ranking: TF-IDF cosine similarity
        - TF-IDF = (term frequency * inverse document frequency) / document norm
        - IDF = log10(N / document_frequency)
        - Document norm normalizes for document length
        
        Args:
            query: Search query string
        
        Returns:
            List of top 100 tuples (doc_id, title) sorted by TF-IDF score (descending)
        """
        self.load_metadata(); self.load_body_index()
        tokens = self.tokenize(query); scores = Counter()
        N = len(self.doc_norms) if self.doc_norms else 6000000  # Total number of documents
        
        # Calculate TF-IDF score for each document
        for t in tokens:
            if self.body_index and t in self.body_index.df:
                idf = math.log(N/self.body_index.df[t], 10)
                for d, tf in self._get_posting(self.body_index, t):
                    di = int(d)
                    # Accumulate normalized TF-IDF contribution from each term
                    scores[di] += (tf*idf)/self.doc_norms.get(di, 1)
        
        res = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(str(d), self.id_to_title.get(int(d), "Unknown")) for d, _ in res[:100]]

    def get_pagerank(self, wiki_ids):
        """
        Get PageRank scores for a list of Wikipedia article IDs.
        PageRank measures article importance based on the link graph structure.
        
        Args:
            wiki_ids: List of document IDs
        
        Returns:
            List of PageRank scores (same order as input), 0 if not found
        """
        self.load_metadata()
        return [self.page_rank.get(int(d), 0) for d in wiki_ids]

    def get_pageview(self, wiki_ids):
        """
        Get page view counts for a list of Wikipedia article IDs.
        Page views indicate article popularity/traffic.
        
        Args:
            wiki_ids: List of document IDs
        
        Returns:
            List of page view counts (same order as input), 0 if not found
        """
        self.load_metadata()
        return [self.page_views.get(int(d), 0) for d in wiki_ids]

# Flask application setup
app = Flask(__name__)
frontend = SearchFrontend()

# API Endpoints

@app.route("/search")
def search():
    """Hybrid search endpoint (title + body). GET parameter: query"""
    return jsonify(frontend.search(request.args.get('query', '')))

@app.route("/search_body")
def search_body():
    """Body-only search endpoint using TF-IDF. GET parameter: query"""
    return jsonify(frontend.search_body(request.args.get('query', '')))

@app.route("/search_title")
def search_title():
    """Title-only search endpoint. GET parameter: query"""
    return jsonify(frontend.search_title(request.args.get('query', '')))

@app.route("/search_anchor")
def search_anchor():
    """Anchor text search endpoint. GET parameter: query"""
    return jsonify(frontend.search_anchor(request.args.get('query', '')))

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    """Get PageRank scores. POST body: JSON list of doc IDs"""
    return jsonify(frontend.get_pagerank(request.get_json() or []))

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    """Get page view counts. POST body: JSON list of doc IDs"""
    return jsonify(frontend.get_pageview(request.get_json() or []))
