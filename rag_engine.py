
import os
import time
from xml.parsers.expat import model

import fitz  # pip install pymupdf

import google.generativeai as genai  # pip install google-generativeai
import torch
import torchvision
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import chromadb  
import pytesseract
from pdf2image import convert_from_path
import re
import unicodedata
from langdetect import detect
from PIL import Image

# =============================================================================
# STEP 0 — Configuration  (change these if you want)
# =============================================================================

CHUNK_SIZE    = 500    # how many words per chunk
CHUNK_OVERLAP = 100    # how many words to repeat between chunks (so context isn't cut off)
RETRIEVAL_K = 20
FINAL_K = 5    # how many chunks to retrieve for each question
DB_PATH       = "./chroma_db"   # folder where ChromaDB saves data
COLLECTION    = "my_pdfs_03"
       # name of the collection inside ChromaDB
EMBEDDING_MODEL = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

RERANKER = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# =============================================================================
# STEP 1 — Connect to Gemini
# =============================================================================

def setup_gemini(api_key: str):
    """
    Give Gemini your API key so we can use it.
    Get a FREE key at: https://aistudio.google.com/app/apikey
    """
    genai.configure(api_key=api_key)
    print("✅ Gemini connected!")


# =============================================================================
# STEP 2 — Connect to ChromaDB  (our free local vector database)
# =============================================================================

def get_collection():
    """
    Open (or create) the ChromaDB collection.
    ChromaDB stores everything in the DB_PATH folder on your computer.
    """
    client = chromadb.PersistentClient(path=DB_PATH)
    # cosine = find chunks whose meaning is closest to the question
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


# =============================================================================
# STEP 3 — Read text from a PDF
# =============================================================================



def clean_text(text: str) -> str:

    lines = text.splitlines()

    # Remove first and last line
    if len(lines) > 4:
        lines = lines[1:-1]

    text = "\n".join(lines)

    # Remove page numbers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    return text

def normalize_text(text):

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()

def read_pdf(pdf_path: str) -> list[dict]:
    """
    Read PDF using:

    1. Native text extraction
    2. Embedded image OCR
    3. Full-page OCR fallback
    4. Text cleaning
    5. Language detection
    """

    pages = []

    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc, start=1):

        # -------------------------------------------------
        # Native text extraction
        # -------------------------------------------------
        text = page.get_text("text")

        # -------------------------------------------------
        # OCR embedded images
        # -------------------------------------------------
        images = page.get_images(full=True)

        for img in images:

            try:

                xref = img[0]

                base_image = doc.extract_image(
                    xref
                )

                image_bytes = base_image["image"]

                image_file = (
                    f"img_{page_num}_{xref}.png"
                )

                with open(
                    image_file,
                    "wb"
                ) as f:
                    f.write(image_bytes)

                image_text = (
                    pytesseract.image_to_string(
                        Image.open(image_file),
                        lang="eng"
                    )
                )

                text += "\n" + image_text

                if os.path.exists(image_file):
                    os.remove(image_file)

            except Exception as e:

                print(
                    f"⚠️ Image OCR failed "
                    f"on page {page_num}: {e}"
                )


        text = clean_text(text)

        text = normalize_text(text)

        text = text.strip()


        if len(text) < 30:

            print(
                f"🔍 Page {page_num}: "
                f"running OCR..."
            )

            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2),
                alpha=False
            )

            temp_img = (
                f"temp_page_{page_num}.png"
            )

            pix.save(temp_img)

            try:

                text = (
                    pytesseract.image_to_string(
                        Image.open(temp_img),
                        lang="eng"
                    )
                )

                text = clean_text(text)

                text = normalize_text(text)

                text = text.strip()

            except Exception as e:

                print(
                    f"❌ OCR failed on page "
                    f"{page_num}: {e}"
                )

                text = ""

            if os.path.exists(temp_img):
                os.remove(temp_img)


        # Skip empty pages

        if len(text) < 30:
            continue

        # Language detection

        try:

            language = detect(text)

        except Exception:

            language = "unknown"

        pages.append({

            "page": page_num,

            "text": text,

            "filename": os.path.basename(
                pdf_path
            ),

            "language": language
        })

    doc.close()

    print(
        f"📄 Read {len(pages)} pages "
        f"from '{os.path.basename(pdf_path)}'"
    )

    return pages

# STEP 4 — Split pages into small chunks

def split_into_chunks(pages: list[dict]) -> list[dict]:
    """
    Take all pages and cut the text into overlapping chunks.

    Why chunks?
      - Gemini has a limit on how much text it can embed at once
      - Smaller chunks = more precise retrieval
      - Overlap = we don't lose context at chunk boundaries

    Returns a list like:
      [ {"chunk_id": "...", "text": "...", "filename": "...", "page": 5}, ... ]
    """
    chunks = []
    chunk_index = 0

    for page_info in pages:
        words = page_info["text"].split()   # split text into individual words

        # slide a window of CHUNK_SIZE words, moving CHUNK_SIZE-CHUNK_OVERLAP each step
        step = CHUNK_SIZE - CHUNK_OVERLAP
        i = 0
        while i < len(words):
            window_words = words[i : i + CHUNK_SIZE]
            chunk_text   = " ".join(window_words)

            if len(chunk_text.strip()) < 50:   # skip tiny leftover bits
                i += step
                continue

            chunks.append({
    "chunk_id": f"{page_info['filename']}_chunk_{chunk_index}",
    "text": chunk_text,
    "filename": page_info["filename"],
    "page": page_info["page"],
    "language": page_info["language"]
})
            chunk_index += 1
            i += step

    print(f"  ✂️  Created {len(chunks)} chunks total")
    return chunks



# STEP 5 — Embed chunks and save to ChromaDB


def embed_and_store(chunks: list[dict], collection) -> int:
    """
    For each chunk:
      1. Ask Gemini to turn the text into a list of numbers (an embedding)
      2. Save that embedding + the original text + metadata into ChromaDB

    ChromaDB saves everything to disk so we only do this ONCE per PDF.
    Next time you start the app, the data is already there.

    Returns the number of NEW chunks added.
    """
    # Which chunk IDs are already in the database?
    existing_ids = set(collection.get(include=[])["ids"])

    # Only process chunks we haven't seen before
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        print("  ⏭️  All chunks already in database, skipping.")
        return 0

    print(f"  🔢 Embedding {len(new_chunks)} new chunks (this may take a moment)...")

    BATCH = 50   # embed 50 at a time to avoid API rate limits
    added = 0

    for i in range(0, len(new_chunks), BATCH):
        batch = new_chunks[i : i + BATCH]
        texts = [c["text"] for c in batch]

        # Ask Gemini for embeddings
        # result = genai.embed_content(
        #     model="models/gemini-embedding-001",
        #     content=texts,
        #     task_type="RETRIEVAL_DOCUMENT",
        # )
        


        embeddings = EMBEDDING_MODEL.encode(
    texts,
    normalize_embeddings=True
)
        # embeddings = result["embedding"]   # list of 768-number vectors

        # Save to ChromaDB
        collection.upsert(
            ids        = [c["chunk_id"] for c in batch],
            embeddings = embeddings,
            documents  = texts,
            metadatas  = [
                {"filename": c["filename"], "page": c["page"],"language": c["language"]}
                for c in batch
            ],
        )
        added += len(batch)
        time.sleep(0.1)   # be polite to the API

    print(f"  ✅ Stored {added} new chunks in ChromaDB")
    return added


# STEP 6 — Ingest a PDF  (combines Steps 3 + 4 + 5)


def ingest_pdf(pdf_path: str, collection) -> dict:
    """
    Full pipeline for one PDF:
      read → chunk → embed → store

    Returns a summary dict with stats.
    """
    print(f"\n📥 Ingesting: {pdf_path}")

    pages  = read_pdf(pdf_path)
    chunks = split_into_chunks(pages)
    added  = embed_and_store(chunks, collection)

    return {
        "filename":   os.path.basename(pdf_path),
        "pages":      len(pages),
        "chunks":     len(chunks),
        "new_chunks": added,
        "status":     "success",
    }


# STEP 7 — Answer a question  (the RAG part)


def ask(question: str, collection, api_key: str) -> dict:
    """
    Full RAG query pipeline:
      1. Embed the question
      2. Find similar chunks in ChromaDB
      3. Build a prompt with those chunks
      4. Ask Gemini to answer
      5. Return the answer + sources

    Returns a dict:
      {
        "answer":  "The answer text...",
        "sources": [ {"filename": "doc.pdf", "page": 3, "snippet": "..."}, ... ],
        "latency_ms": 1234,
      }
    """
    if collection.count() == 0:
        return {
            "answer":     "No documents have been indexed yet. Please upload some PDFs first.",
            "sources":    [],
            "latency_ms": 0,
        }

    t_start = time.time()

    # ── 7a. Embed the question ─────────────────────────────────────────────────
    q_embedding = EMBEDDING_MODEL.encode(
    [question],
    normalize_embeddings=True
)[0]

    # ── 7b. Find the most similar chunks ──────────────────────────────────────
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=min(RETRIEVAL_K, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    retrieved_docs  = results["documents"][0]    # list of chunk texts
    retrieved_metas = results["metadatas"][0]    # list of {filename, page}
    retrieved_dists = results["distances"][0]    # cosine distance (lower = better)
    pairs = [
        (question, doc)
        for doc in retrieved_docs
    ]
    rerank_scores = RERANKER.predict(pairs)

    ranked_results = sorted(
        zip(
            retrieved_docs,
            retrieved_metas,
            retrieved_dists,
            rerank_scores
        ),
        key=lambda x: x[3],
        reverse=True
    )

    ranked_results = ranked_results[:FINAL_K]

    retrieved_docs = [x[0] for x in ranked_results]
    retrieved_metas = [x[1] for x in ranked_results]
    retrieved_dists = [x[2] for x in ranked_results]
    # ── 7c. Build the context block for Gemini ─────────────────────────────────
    context_parts = []
    for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metas), start=1):
        context_parts.append(
            f"[Excerpt {i}]  Source: {meta['filename']}  |  Page {meta['page']}\n{doc}"
        )
    context_text = "\n\n".join(context_parts)

    # ── 7d. Build the full prompt ──────────────────────────────────────────────
    prompt = f"""You are a helpful research assistant.
Answer the question below using ONLY the provided excerpts.
Always mention which PDF and page number your answer comes from.
If the excerpts don't contain enough information, say so clearly.

---EXCERPTS---
{context_text}
---END EXCERPTS---

Question: {question}

Answer:"""

    # ── 7e. Call Gemini to generate the answer ────────────────────────────────
    model    = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    answer   = response.text

    # ── 7f. Build the sources list ─────────────────────────────────────────────
    sources = []
    seen    = set()
    for doc, meta, dist in zip(retrieved_docs, retrieved_metas, retrieved_dists):
        key = (meta["filename"], meta["page"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "filename":  meta["filename"],
                "page":      meta["page"],
                "vector_score": round((1 - dist) * 100, 1),   # % relevance score
                "snippet":   doc[:200] + "…" if len(doc) > 200 else doc,
            })

    latency_ms = int((time.time() - t_start) * 1000)

    return {
    "answer": answer,
    "sources": sources,
    "retrieved_chunks": [
        {
            "page": meta["page"],
            "file": meta["filename"],
            "score": round((1 - dist) * 100, 2),
            "chunk": doc[:300]
        }
        for doc, meta, dist in zip(
            retrieved_docs,
            retrieved_metas,
            retrieved_dists
        )
    ],
    "latency_ms": latency_ms
}



# STEP 8 — Helper: how many documents are indexed?


def get_stats(collection) -> dict:
    """Return basic stats about what's in the database."""
    total_chunks = collection.count()

    if total_chunks == 0:
        return {"total_chunks": 0, "total_documents": 0, "documents": []}

    all_meta  = collection.get(include=["metadatas"])["metadatas"]
    filenames = sorted(set(m["filename"] for m in all_meta))

    return {
        "total_chunks":    total_chunks,
        "total_documents": len(filenames),
        "documents":       filenames,
    }
