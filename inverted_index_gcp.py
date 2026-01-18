# Inverted Index Implementation for Google Cloud Platform
# This module implements an inverted index data structure optimized for large-scale
# Wikipedia data storage and retrieval on Google Cloud Storage (GCS).
# Key features:
# - Split storage across multiple binary files to handle large datasets
# - Efficient binary encoding of posting lists (doc_id, term_frequency pairs)
# - Support for both local filesystem and GCS bucket storage
# - Memory-efficient streaming reads/writes

import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
from time import time
from pathlib import Path
import pickle
from google.cloud import storage
from collections import defaultdict
from contextlib import closing

# Google Cloud Project ID
PROJECT_ID = 'warm-skill-481016-r0'

def get_bucket(bucket_name):
    """
    Get a reference to a Google Cloud Storage bucket.
    
    Args:
        bucket_name: Name of the GCS bucket
    
    Returns:
        storage.Bucket object
    """
    return storage.Client(PROJECT_ID).bucket(bucket_name)

def _open(path, mode, bucket=None):
    """
    Universal file opener that works with both local filesystem and GCS.
    
    Args:
        path: File path (local or GCS)
        mode: File open mode ('rb', 'wb', etc.)
        bucket: GCS bucket object (None for local files)
    
    Returns:
        File-like object (either standard file or GCS blob)
    """
    if bucket is None:
        return open(path, mode)
    return bucket.blob(path).open(mode)

# Maximum size for each binary file (approximately 2MB)
# When a file reaches this size, a new file is created
# This prevents individual files from becoming too large
BLOCK_SIZE = 1999998

class MultiFileWriter:
    """
    Sequential binary writer to multiple files of up to BLOCK_SIZE each.
    
    Automatically creates new files when the current file reaches BLOCK_SIZE.
    Files are named with sequential numbers: {name}_000.bin, {name}_001.bin, etc.
    Tracks the location of each write operation for later retrieval.
    
    Usage:
        writer = MultiFileWriter('output_dir', 'postings', 'my-bucket')
        locations = writer.write(binary_data)
        writer.close()
    """
    
    def __init__(self, base_dir, name, bucket_name=None):
        """
        Initialize the multi-file writer.
        
        Args:
            base_dir: Directory path for output files
            name: Base name for the files (will be suffixed with _000, _001, etc.)
            bucket_name: Optional GCS bucket name (None for local files)
        """
        self._base_dir = Path(base_dir)
        self._name = name
        self._bucket = None if bucket_name is None else get_bucket(bucket_name)
        # Generator that creates files with sequential numbering
        self._file_gen = (_open(str(self._base_dir / f'{name}_{i:03}.bin'), 
                                'wb', self._bucket) 
                          for i in itertools.count())
        self._f = next(self._file_gen)
           
    def write(self, b):
        """
        Write binary data, automatically spanning multiple files if needed.
        
        If the data doesn't fit in the current file (would exceed BLOCK_SIZE),
        it's split across multiple files. Tracks the exact location of each chunk.
        
        Args:
            b: Binary data (bytes) to write
        
        Returns:
            List of (filename, offset) tuples indicating where each chunk was written
        """
        locs = []
        while len(b) > 0:
            pos = self._f.tell()  # Current position in file
            remaining = BLOCK_SIZE - pos  # Space left in current file
            
            # If current file is full, create a new one
            if remaining == 0:  
                self._f.close()
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            
            # Write what fits in current file
            self._f.write(b[:remaining])
            name = self._f.name if hasattr(self._f, 'name') else self._f._blob.name
            locs.append((name, pos))
            b = b[remaining:]  # Continue with remaining data
        return locs

    def close(self):
        """Close the current file. Always call this when done writing."""
        self._f.close()

class MultiFileReader:
    """
    Sequential binary reader of multiple files of up to BLOCK_SIZE each.
    
    Reads binary data that was written by MultiFileWriter, potentially spanning
    multiple files. Maintains a cache of open file handles for efficiency.
    
    Usage:
        reader = MultiFileReader('input_dir', 'my-bucket')
        data = reader.read(locations, num_bytes)
        reader.close()
    """
    
    def __init__(self, base_dir, bucket_name=None):
        """
        Initialize the multi-file reader.
        
        Args:
            base_dir: Directory path containing the files to read
            bucket_name: Optional GCS bucket name (None for local files)
        """
        self._base_dir = Path(base_dir)
        self._bucket = None if bucket_name is None else get_bucket(bucket_name)
        self._open_files = {}  # Cache of open file handles

    def read(self, locs, n_bytes):
        """
        Read binary data from specified locations, potentially spanning multiple files.
        
        Args:
            locs: List of (filename, offset) tuples from MultiFileWriter.write()
            n_bytes: Total number of bytes to read
        
        Returns:
            Binary data (bytes) concatenated from all locations
        """
        b = []
        for f_name, offset in locs:
            # Support both full paths and relative paths (extract basename)
            f_name_str = str(self._base_dir / os.path.basename(f_name))
            
            # Open file if not already in cache
            if f_name_str not in self._open_files:
                self._open_files[f_name_str] = _open(f_name_str, 'rb', self._bucket)
            
            f = self._open_files[f_name_str]
            f.seek(offset)  # Move to the specified position
            n_read = min(n_bytes, BLOCK_SIZE - offset)  # Read up to end of block
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b"".join(b)  # Concatenate all chunks
  
    def close(self):
        """Close all open file handles. Always call this when done reading."""
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager support: automatically close files when exiting 'with' block."""
        self.close()
        return False 

# Binary encoding constants for posting lists
# Each posting list entry is a (doc_id, term_frequency) pair
TUPLE_SIZE = 6       # 6 bytes per entry: 4 bytes for doc_id + 2 bytes for term_frequency
TF_MASK = 2 ** 16 - 1  # Mask to extract term_frequency (16-bit value) 

class InvertedIndex:
    """
    Inverted Index data structure for efficient term-to-document lookups.
    
    The inverted index maps each term to a posting list of (doc_id, term_frequency) pairs.
    Designed to handle large-scale data by storing posting lists in binary files.
    
    Key attributes:
    - df: Document frequency (number of documents containing each term)
    - term_total: Total occurrences of each term across all documents
    - _posting_list: In-memory posting lists (used during construction, not saved)
    - posting_locs: File locations of posting lists (for retrieval)
    
    Binary encoding:
    - Each posting is 6 bytes: 4 bytes doc_id + 2 bytes term_frequency
    - Doc_id is shifted left 16 bits, term_frequency is in lower 16 bits
    """
    
    def __init__(self, docs={}):
        """
        Initialize the inverted index.
        
        Args:
            docs: Dictionary mapping doc_id -> list of tokens (optional)
        """
        self.df = Counter()  # Document frequency for each term
        self.term_total = Counter()  # Total occurrences of each term
        self._posting_list = defaultdict(list)  # In-memory posting lists
        self.posting_locs = defaultdict(list)  # File locations of posting lists

        # Build index from provided documents
        for doc_id, tokens in docs.items():
            self.add_doc(doc_id, tokens)

    def add_doc(self, doc_id, tokens):
        """
        Add a document to the inverted index.
        
        Process:
        1. Count term frequencies in the document
        2. Update global term statistics (df, term_total)
        3. Add (doc_id, tf) to each term's posting list
        
        Args:
            doc_id: Unique document identifier
            tokens: List of terms/tokens in the document
        """
        w2cnt = Counter(tokens)  # Count occurrences of each term
        self.term_total.update(w2cnt)  # Update total counts
        
        # Add document to posting list of each term
        for w, cnt in w2cnt.items():
            self.df[w] = self.df.get(w, 0) + 1  # Increment document frequency
            self._posting_list[w].append((doc_id, cnt))  # Add to posting list

    def write_index(self, base_dir, name, bucket_name=None):
        """
        Write the inverted index to storage.
        Currently only writes metadata (df, term_total, posting_locs).
        Posting lists are written separately using write_a_posting_list().
        
        Args:
            base_dir: Directory path for output
            name: Base name for the index file
            bucket_name: Optional GCS bucket name
        """
        self._write_globals(base_dir, name, bucket_name)

    def _write_globals(self, base_dir, name, bucket_name):
        """
        Write the index metadata to a pickle file.
        Uses __getstate__ to exclude _posting_list (too large for memory).
        
        Args:
            base_dir: Directory path for output
            name: Base name for the pickle file
            bucket_name: Optional GCS bucket name
        """
        path = str(Path(base_dir) / f'{name}.pkl')
        bucket = None if bucket_name is None else get_bucket(bucket_name)
        with _open(path, 'wb', bucket) as f:
            pickle.dump(self, f)

    def __getstate__(self):
        """
        Control what gets pickled when saving the index.
        Excludes _posting_list to save memory (posting lists are stored in binary files).
        Only saves metadata: df, term_total, posting_locs.
        
        Returns:
            Dictionary of attributes to pickle (without _posting_list)
        """
        state = self.__dict__.copy()
        del state['_posting_list']  # Don't save in-memory posting lists
        return state

    def posting_lists_iter(self, base_dir, bucket_name=None):
        """
        Iterator over all posting lists in the index.
        Reads binary data from files and decodes it into (doc_id, tf) pairs.
        
        Binary decoding:
        - Each entry is 6 bytes
        - First 4 bytes: doc_id (big-endian integer)
        - Next 2 bytes: term_frequency (big-endian integer)
        
        Args:
            base_dir: Directory containing the posting list files
            bucket_name: Optional GCS bucket name
        
        Yields:
            Tuples of (term, posting_list) where posting_list is [(doc_id, tf), ...]
        """
        with closing(MultiFileReader(base_dir, bucket_name)) as reader:
            for w, locs in self.posting_locs.items():
                # Read binary data for this term's posting list
                b = reader.read(locs, self.df[w] * TUPLE_SIZE)
                posting_list = []
                
                # Decode each (doc_id, tf) pair
                for i in range(self.df[w]):
                    doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
                    tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
                    posting_list.append((doc_id, tf))
                
                yield w, posting_list

    def read_a_posting_list(self, base_dir, w, bucket_name=None):
        """
        Read the posting list for a specific term.
        
        This is the primary method for query processing - retrieves all documents
        containing a given term along with their term frequencies.
        
        Binary decoding:
        - Each entry is 6 bytes: 4 bytes doc_id + 2 bytes term_frequency
        
        Args:
            base_dir: Directory containing the posting list files
            w: Term to look up
            bucket_name: Optional GCS bucket name (CRITICAL: must be passed to reader)
        
        Returns:
            List of (doc_id, term_frequency) tuples, sorted by doc_id
            Returns empty list if term not found
        """
        posting_list = []
        if not w in self.posting_locs:
            return posting_list
        
        # CRITICAL FIX: Pass bucket_name to MultiFileReader for GCS access
        with closing(MultiFileReader(base_dir, bucket_name)) as reader:
            locs = self.posting_locs[w]
            b = reader.read(locs, self.df[w] * TUPLE_SIZE)
            
            # Decode binary data into (doc_id, tf) pairs
            for i in range(self.df[w]):
                doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
                tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
        
        return posting_list

    @staticmethod
    def write_a_posting_list(b_w_pl, base_dir, bucket_name=None):
        """
        Write posting lists to binary files (static method for parallel processing).
        
        Binary encoding:
        - Each (doc_id, tf) pair is encoded as 6 bytes
        - doc_id is shifted left 16 bits, tf occupies lower 16 bits
        - Format: (doc_id << 16 | tf).to_bytes(6, 'big')
        
        Also saves a separate pickle file with the posting locations
        ({bucket_id}_posting_locs.pickle) for later retrieval.
        
        Args:
            b_w_pl: Tuple of (bucket_id, list_of_(term, posting_list)_pairs)
            base_dir: Directory for output files
            bucket_name: Optional GCS bucket name
        
        Returns:
            bucket_id (for tracking completion)
        """
        posting_locs = defaultdict(list)
        bucket_id, list_w_pl = b_w_pl
        
        with closing(MultiFileWriter(base_dir, bucket_id, bucket_name)) as writer:
            # Encode and write each posting list
            for w, pl in list_w_pl:
                # Encode: doc_id in upper 4 bytes, tf in lower 2 bytes
                b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                              for doc_id, tf in pl])
                locs = writer.write(b)
                posting_locs[w].extend(locs)
            
            # Save the posting locations for later retrieval
            path = str(Path(base_dir) / f'{bucket_id}_posting_locs.pickle')
            bucket = None if bucket_name is None else get_bucket(bucket_name)
            with _open(path, 'wb', bucket) as f:
                pickle.dump(posting_locs, f)
        
        return bucket_id

    @staticmethod
    def read_index(base_dir, name, bucket_name=None):
        """
        Load an inverted index from a pickle file.
        
        This restores the index metadata (df, term_total, posting_locs) that was
        saved with write_index(). The actual posting lists remain in binary files
        and are loaded on-demand using read_a_posting_list().
        
        Args:
            base_dir: Directory containing the index files
            name: Base name of the index file (without .pkl extension)
            bucket_name: Optional GCS bucket name
        
        Returns:
            InvertedIndex object with metadata loaded
        """
        path = str(Path(base_dir) / f'{name}.pkl')
        bucket = None if bucket_name is None else get_bucket(bucket_name)
        with _open(path, 'rb', bucket) as f:
            return pickle.load(f)