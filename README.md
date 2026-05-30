# RAG Pipeline Overview

## 1. PDF Ingestion & Preprocessing

The system ingests PDF documents using PyMuPDF. It supports:

* Native text extraction from digital PDFs
* OCR for scanned PDF pages using Tesseract
* OCR for embedded images and screenshots within PDFs
* Text cleaning (header/footer removal and page number removal)
* Text normalization for consistent embedding generation
* Language detection for multilingual document support

This ensures both digital and scanned PDFs become searchable within the knowledge base.

---

## 2. Chunking

Extracted text is split into overlapping chunks using a sliding window approach.

Configuration:

* Chunk Size: 500 words
* Chunk Overlap: 100 words

Benefits:

* Improves retrieval precision
* Preserves context across chunk boundaries
* Produces deterministic chunks for reproducibility

Each chunk stores metadata including:

* Filename
* Page Number
* Language

---

## 3. Embedding Generation

The system uses the open-source embedding model:

BAAI/bge-base-en-v1.5

Purpose:

* Convert document chunks into dense vector representations
* Enable semantic similarity search

Embeddings are generated once during ingestion and stored permanently in the vector database.

---

## 4. Vector Database

ChromaDB is used as the vector database.

Features:

* Persistent storage
* HNSW-based Approximate Nearest Neighbor (ANN) search
* Fast semantic retrieval at scale

Stored data includes:

* Embeddings
* Original chunk text
* Metadata

---

## 5. Retrieval

When a user submits a query:

1. The question is embedded using the same BGE model.
2. ChromaDB retrieves the Top-K most relevant chunks using cosine similarity.
3. Metadata and similarity scores are returned.

---

## 6. Reranking

A Cross Encoder reranker is used:

cross-encoder/ms-marco-MiniLM-L-6-v2

Purpose:

* Improve retrieval quality
* Reorder retrieved chunks based on query-document relevance
* Increase answer accuracy

Pipeline:

Question → Top 20 Retrieval → Cross Encoder Reranking → Top 5 Context Chunks

---

## 7. Answer Generation

The final context is passed to Gemini 2.5 Flash.

The model:

* Uses only retrieved evidence
* Generates grounded answers
* Includes source citations
* References PDF filenames and page numbers

---

## 8. Retrieval Visualization

For transparency and explainability, the system returns:

* Retrieved chunks
* Similarity scores
* Source PDF names
* Source page numbers

This allows users to inspect the evidence used for answer generation.

---

## 9. Evaluation Metrics

The system is designed to support evaluation of:

* Latency
* Citation Accuracy

These metrics help measure retrieval quality, response speed, and answer reliability.

---

## End-to-End Flow

PDFs
↓
Text Extraction + OCR
↓
Cleaning & Normalization
↓
Chunking
↓
BGE Embeddings
↓
ChromaDB Storage
↓
User Question
↓
Question Embedding
↓
ANN Retrieval
↓
Cross Encoder Reranking
↓
Gemini Answer Generation
↓
Answer + Citations + Retrieved Chunks
