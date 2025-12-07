# Multilingual-Neural-Search-Engine

## Project Summary

### Description: 
Many organizations have documents in multiple languages (e.g. English, Hindi, Telugu, Spanish, etc.). A user might query in one language (say Hindi) but relevant documents might exist in another (say English). Traditional search (keyword-based, per language) fails: you either need translation or manual indexing per language.

You want a single search engine where the user query (in any supported language) returns semantically relevant documents across languages — based on meaning, not keyword match.

This is especially useful for multinational corpora, multilingual FAQs, cross-lingual knowledgebases, etc

So, this is a Multilingual Neural Search Engine that encodes questions into semantic embeddings and performs fast similarity search with FAISS. This supports queries in many languages using a multilingual sentence-transformer.
### Goal:
Return relevant results even when queries are issued in different languages than the indexed data.

## Features

- Multilingual embeddings: Uses sentence-transformers (multilingual) for language-agnostic embeddings.
- Fast retrieval: FAISS index with cosine similarity for efficient search.
- Batch encoding: Faster indexing via batched embedding generation.
- Language detection: Detects query language for logging and diagnostics.
- Interactive search helper: multilingual_search(query, top_k, threshold) function.

## Quick Start

- File to run: sample.ipynb
- Dataset: data_train.csv (place in repository root)