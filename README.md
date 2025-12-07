# Multilingual-Neural-Search-Engine

## Project Summary

### Description: 
Multilingual Neural Search Engine that encodes questions into semantic embeddings and performs fast similarity search with FAISS. Supports queries in many languages using a multilingual sentence-transformer.
### Goal:
Return relevant results even when queries are issued in different languages than the indexed data.

## Features

- Multilingual embeddings: Uses sentence-transformers (multilingual) for language-agnostic embeddings.
- Fast retrieval: FAISS index with cosine similarity for efficient search.
- Batch encoding: Faster indexing via batched embedding generation.
- Language detection: Detects query language for logging and diagnostics.
- Interactive search helper: multilingual_search(query, top_k, threshold) function.

## Quick Start

File to run: sample.ipynb
Dataset: data_train.csv (place in repository root)