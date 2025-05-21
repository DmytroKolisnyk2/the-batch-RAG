# Batch Newsletter Multimodal RAG System

This repository implements a multimodal Retrieval-Augmented Generation (RAG) system for "The Batch" AI newsletter from deeplearning.ai. We scrape text and images directly from the newsletter's, chunk and index content in Qdrant, and provide a Streamlit demo for interactive querying.

---

## Table of Contents

1. [Setup](#setup)
2. [Project Structure](#project-structure)
3. [Environment Variables](#environment-variables)
4. [Data Collection Pipeline](#data-collection-pipeline)
5. [Content Processing Pipeline](#content-processing-pipeline)
6. [Indexing into Qdrant](#indexing-into-qdrant)
7. [RAG Module](#rag-module)
8. [Streamlit Demo](#streamlit-demo)
9. [Approach & Reasoning](#approach--reasoning)
10. [Tools & Models](#tools--models)

---

## Setup

1. **Create a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install and configure Ngrok (for remote Streamlit demo)**

   ```bash
   pip install pyngrok
   ngrok authtoken YOUR_NGROK_TOKEN
   ```
---

## Project Structure

```
├── data/
│   ├── raw/
│   │   ├── pages/            # JSON index pages from Next.js
│   │   └── posts/            # Raw post JSONs (slug.json)
│   └── processed/
│       ├── chunks.jsonl      # Text chunks for RAG
│       └── images.jsonl      # Image metadata & captions
├── rag_module.py            # Core RAG logic (Qdrant queries + LL model)
├── streamlit_app.py         # Streamlit UI for interactive queries
├── The_Batch_Multimodal_RAG_System.ipynb  # Colab notebook for full pipeline
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Environment Variables

| Variable         | Description                                           |
| ---------------- | ----------------------------------------------------- |
| `QDRANT_URL`     | URL for your Qdrant Cloud or local instance           |
| `QDRANT_API_KEY` | API key for Qdrant Cloud                              |
| `OPENAI_API_KEY` | API key for OpenAI                                    |
| `NGROK_TOKEN`    | Authtoken for Ngrok (for exposing Streamlit remotely) |

Export these in your shell or set them in your environment before running scripts.

---

## Data Collection Pipeline

1. **Fetch Next.js build ID**

   * The Batch newsletter frontend is built with Next.js.
   * We request `https://www.deeplearning.ai/the-batch` and parse `#__NEXT_DATA__` to extract the current `buildId`.
   * This allows us to directly fetch JSON pages without parsing HTML.

2. **Download paginated index JSON**

   * Loop through pages: `https://www.deeplearning.ai/_next/data/{buildId}/the-batch/page/{page}.json`.
   * Stop when a 404 response is encountered.
   * Save index JSON in `data/raw/pages/`.

3. **Extract post metadata**

   * Aggregate all post entries into `data/posts_meta.jsonl` for reference.

4. **Download each post JSON**

   * For each post `slug`, request:

     ```
     GET https://www.deeplearning.ai/_next/data/{buildId}/the-batch/{slug}.json
     ```
   * Save under `data/raw/posts/{slug}.json`.
   * Handle rate limits (HTTP 429/504) with up to 3 retries using exponential backoff.
   * **Estimated Runtime:** \~1–3 minutes to scrape \~100 posts (varies based on network and rate limits).

---

## Content Processing Pipeline

1. **Parse HTML to Markdown & extract images**

   * Use `BeautifulSoup` to remove scripts/styles and extract `<img>` tags.
   * Convert HTML body to Markdown using `html2text`.

2. **Chunk text**

   * Split Markdown into overlapping chunks (`chunk_size=768`, `overlap=128`) via LangChain's `RecursiveCharacterTextSplitter`.
   * Each chunk becomes a `Document` with metadata: `slug`, `title`, `published`, `chunk_id`, `n_chunks`.

3. **Extract image metadata**

   * For each `<img>`, record `url` and `alt` text.
   * Prepend feature images if present.

4. **Serialize outputs**

   * Write text chunks to `data/processed/chunks.jsonl`.
   * Write image records to `data/processed/images.jsonl`.

---

## Indexing into Qdrant

1. **Configure collection**

   * Collection name: `batch_multimodal`
   * Vector dimensions: `text` (768), `image` (512)
   * Distance metric: Cosine

2. **Text indexing**

   * Load sentence embeddings via `intfloat/e5-base-v2`.
   * Batch-encode chunks, upsert with payload (`page_content`, metadata).

3. **Image indexing**

   * Generate captions with BLIP (`Salesforce/blip-image-captioning-base`).
   * Fallback to `alt` text when available.
   * Encode captions with CLIP (`clip-ViT-B-32`), upsert with payload (`img_url`, `caption`).

---

## RAG Module

* **File**: `rag_module.py`

* **Key Components**:

  1. **`_search_multimodal(query, k)`**:

     * Encodes the user query into two vector spaces:

       * **Text**: via `HuggingFaceEmbeddings` (`intfloat/e5-base-v2`).
       * **Image**: via `SentenceTransformer` (`clip-ViT-B-32`).
     * Issues `query_points` calls to Qdrant for each modality with optimized search parameters (`hnsw_ef=256`, `score_threshold` for images).
     * Retrieves \~2×K hits per modality, then **deduplicates** by `slug`, preferring higher-scored text hits and filling with image hits up to K.
  2. **`_build_context(hits)`**:

     * Iterates over retrieved `ScoredPoint` payloads:

       * **Text payloads**: formats each chunk as `### Title` followed by the text.
       * **Image payloads**: inserts placeholder `[Image: caption]` blocks.
     * Collects detailed `passages` (title, chunk, slug) and `images` (url, caption) lists.
     * Concatenates into a single **LLM prompt context**, limited to `RETURN_K` top items to control prompt length.
  3. **`answer_with_rag(question)`**:

     * Calls `_search_multimodal`, then `_build_context` on top hits.
     * Applies the `ChatPromptTemplate` to inject context and question into the LLM prompt.
     * Invokes `ChatOpenAI` (`gpt-4o-mini`) with `temperature=0.2`.
     * Parses the model output via `StrOutputParser` to obtain a clean answer string.
     * Returns a tuple `(answer, passages, images)` ready for downstream rendering.

* **Approaches & Best Practices**:

  * **Multimodal Fusion**: combining text and image embeddings ensures comprehensive retrieval.
  * **Score-based Deduplication**: prevents redundant content by tracking unique `slug` keys.
  * **Prompt Engineering**: clear templating and context chunking maintain LLM coherence and relevance.
  * **Search Parameter Tuning**: using `hnsw_ef=256` balances retrieval speed and recall.
  * **Lightweight Parsing**: `StrOutputParser` enforces structured output without manual string cleaning.

## Streamlit Demo

* **File**: `streamlit_app.py`
* **Usage**:

  ```bash
  streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.fileWatcherType none
  ```
* **Features**:

  * Text input for free-form questions about The Batch.
  * Displays LLM-generated answer, relevant text passages, and images with captions.
  * Built with Streamlit & Qdrant-powered RAG.

---

## Approach & Reasoning

* **Structured Data Ingestion**: Leverage Next.js build ID to fetch raw JSON and handle pagination seamlessly, avoiding brittle HTML parsing and ensuring complete data capture.
* **Multimodal Approach**: Index both text chunks (E5 embeddings) and image captions (BLIP+CLIP) to support richer, cross-modal queries.
* **Context Optimization**: Use overlapping text chunks (768 characters, 128 overlap) to maintain continuity and provide coherent context to the LLM.
* **Efficient Embeddings**: Utilize E5 for fast, high-quality text embeddings and CLIP for semantic image embeddings to balance performance and accuracy.
* **Adaptive Captioning**: Generate image captions on-the-fly with BLIP, falling back to alt text for images lacking descriptive metadata, improving recall in image retrieval.

## Tools & Models

| Stage            | Tool / Library                                 | Model / Version                         |
| ---------------- | ---------------------------------------------- | --------------------------------------- |
| Scraping         | `requests`, `BeautifulSoup`                    |                                         |
| HTML→Markdown    | `html2text`                                    |                                         |
| Chunking         | `langchain_text_splitters`                     |                                         |
| Text Embeddings  | `SentenceTransformer`, `HuggingFaceEmbeddings` | `intfloat/e5-base-v2`                   |
| Image Captioning | `transformers` pipeline (BLIP)                 | `Salesforce/blip-image-captioning-base` |
| Image Embeddings | `SentenceTransformer`                          | `clip-ViT-B-32`                         |
| Vector Store     | `qdrant-client`                                | Qdrant Cloud / Local                    |
| LLM              | `langchain.chat_models.ChatOpenAI`             | `gpt-4o-mini`                           |
| Demo UI          | `streamlit`, `pyngrok`                         |                                         |

---
