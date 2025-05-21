import os
from typing import List, Tuple, Dict

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# ── Configuration ────────────────────────────────────────────────────────────
COLLECTION = "batch_multimodal"
TOP_K = 8
RETURN_K = 4
TEMPERATURE = 0.2
MODEL_NAME = "gpt-4o-mini"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

assert (
    QDRANT_URL and QDRANT_API_KEY
), "Please set QDRANT_URL and QDRANT_API_KEY environment variables"

# ── Initialize clients & models ─────────────────────────────────────────────
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
text_encoder = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
image_encoder = SentenceTransformer("clip-ViT-B-32")
llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)

# ── Prompt template ──────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_template(
    """
You are an AI assistant helping with questions about AI news from "The Batch" newsletter.
Answer the question based only on the provided context.

Context:
{context}

Question: {question}

Answer:
"""
)


def _search_multimodal(query: str, k: int = TOP_K) -> List[qmodels.ScoredPoint]:
    """
    Perform text and image search in Qdrant and return top-k unique hits by slug.
    """
    # Encode query for text and image modalities
    qv_text = text_encoder.embed_query(query)
    qv_image = image_encoder.encode(query).tolist()

    # Search text vectors
    text_hits = client.query_points(
        collection_name=COLLECTION,
        query=qv_text,
        using="text",
        limit=k * 2,
        with_payload=True,
        search_params=qmodels.SearchParams(hnsw_ef=256),
    ).points

    # Search image vectors
    image_hits = client.query_points(
        collection_name=COLLECTION,
        query=qv_image,
        using="image",
        limit=k * 2,
        with_payload=True,
        score_threshold=0.2,
        search_params=qmodels.SearchParams(hnsw_ef=256),
    ).points

    # Deduplicate by slug, preferring text hits
    best: Dict[str, qmodels.ScoredPoint] = {}
    for hit in sorted(text_hits, key=lambda x: -x.score):
        slug = hit.payload.get("slug")
        if slug:
            best.setdefault(slug, hit)
    for hit in sorted(image_hits, key=lambda x: -x.score):
        slug = hit.payload.get("slug")
        if slug:
            best.setdefault(slug, hit)

    # Return top-k by score
    return sorted(best.values(), key=lambda x: -x.score)[:k]


def _build_context(
    hits: List[qmodels.ScoredPoint],
) -> Tuple[str, List[Dict], List[Dict]]:
    """
    Build a combined text context string, plus lists of passages and images.
    """
    text_blocks: List[str] = []
    passages: List[Dict] = []
    images: List[Dict] = []

    for pt in hits:
        payload = pt.payload
        # Text chunk
        if "page_content" in payload:
            title = payload.get("title", "Untitled")
            chunk = payload["page_content"]
            slug = payload["slug"]
            text_blocks.append(f"### {title}\n{chunk}\n")
            passages.append({"title": title, "chunk": chunk, "slug": slug})
        # Image entry
        elif "img_url" in payload:
            caption = payload.get("caption", "")
            url = payload["img_url"]
            text_blocks.append(f"[Image: {caption}]\n")
            images.append({"url": url, "caption": caption})

    context_str = "\n".join(text_blocks)
    return context_str, passages, images


def answer_with_rag(question: str) -> Tuple[str, List[Dict], List[Dict]]:
    """
    Run multimodal search, build context, and generate an answer via the LLM.
    Returns the answer string, list of text passages, and list of images.
    """
    hits = _search_multimodal(question, k=TOP_K)
    context, passages, images = _build_context(hits[:RETURN_K])

    # Format the prompt and call the language model
    response = llm(
        prompt.format_prompt(context=context, question=question).to_messages()
    )
    answer = StrOutputParser().parse(response.content)
    return answer, passages, images
