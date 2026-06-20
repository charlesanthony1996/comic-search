"""
rag.py — Retrieval Augmented Generation for Comic Search
=========================================================
Combines RRF retrieval with Claude (Anthropic API) to answer
natural language questions about your comic dataset.

Instead of just returning page images, RAG generates a full
natural language answer grounded in the actual comic dialogue.

Setup:
    pip install anthropic

    export ANTHROPIC_API_KEY=your_key_here

Usage:
    from rag import rag_answer, rag_pipeline

    # simple answer
    answer = rag_answer("Does Punisher kill Wilson Fisk?")
    print(answer)

    # full pipeline with retrieved pages + answer
    result = rag_pipeline("What happens when Daredevil fights the Kingpin?")
    print(result["answer"])
    print(result["sources"])
"""

import json
import os
import anthropic
import numpy as np

from pathlib import Path

# ── Paths (must match main.py) ─────────────────────────────────────────────────
base_dir    = Path(__file__).parent
image_dir   = base_dir / "dataset"
corpus_path = base_dir / "dataset_text.json"
index_file  = base_dir / "clip_index.npz"


# ── Anthropic client ───────────────────────────────────────────────────────────
def get_client():
    """
    Initialise Anthropic client.
    Reads API key from ANTHROPIC_API_KEY environment variable.
    """
    api_key = os.getenv("os.environ["ANTHROPIC_API_KEY"]")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set.\n"
            "Run: export ANTHROPIC_API_KEY=your_key_here"
        )
    return anthropic.Anthropic(api_key=api_key)


# ── Step 1: Retrieve — reuse your existing RRF search ─────────────────────────
def retrieve(query: str, top_k: int = 5) -> list[str]:
    """
    Retrieve top_k most relevant page filenames using RRF fusion.
    Imports search functions from main.py to avoid code duplication.
    """
    # import here to avoid circular imports
    from main import run_search, bm25_search

    # get results from both retrievers
    clip_results  = run_search(query, top_k=100, show=False) or []
    bm25_raw      = bm25_search(query, top_k=100)
    bm25_results  = [fname for _, fname in bm25_raw]

    # RRF fusion
    k = 60
    rrf_scores = {}
    for rank, fname in enumerate(clip_results, 1):
        rrf_scores[fname] = rrf_scores.get(fname, 0) + 1 / (k + rank)
    for rank, fname in enumerate(bm25_results, 1):
        rrf_scores[fname] = rrf_scores.get(fname, 0) + 1 / (k + rank)

    ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return [fname for fname, _ in ranked[:top_k]]


# ── Step 2: Augment — extract OCR text from retrieved pages ────────────────────
def get_context(filenames: list[str]) -> list[dict]:
    """
    Load OCR text for each retrieved page from dataset_text.json.
    Returns list of {filename, character, text} dicts.

    Pages with no OCR text are included with a note — the LLM
    is told the page exists but has no readable text.
    """
    if not corpus_path.exists():
        raise FileNotFoundError(
            "dataset_text.json not found — run build_text_corpus() first"
        )

    with open(corpus_path) as f:
        corpus = json.load(f)

    context_pages = []

    for fname in filenames:
        # infer character from filename prefix
        char = fname.split("_page_")[0] if "_page_" in fname else "unknown"

        ocr_text = corpus.get(fname, "").strip()

        context_pages.append({
            "filename":  fname,
            "character": char,
            "text":      ocr_text if ocr_text else "[no readable text on this page]",
            "has_text":  bool(ocr_text),
        })

    return context_pages


# ── Step 3: Generate — send context + query to Claude ─────────────────────────
def generate_answer(query: str, context_pages: list[dict],
                    model: str = "claude-sonnet-4-6") -> str:
    """
    Send retrieved page context + query to Claude and get a grounded answer.

    The system prompt instructs Claude to:
    - Only answer based on the provided comic context
    - Cite which pages the information comes from
    - Admit uncertainty if the answer isn't in the retrieved pages
    """
    client = get_client()

    # build context string from retrieved pages
    context_str = ""
    for i, page in enumerate(context_pages, 1):
        context_str += f"\n--- Page {i}: {page['filename']} ({page['character']}) ---\n"
        context_str += page["text"] + "\n"

    # system prompt — instructs the LLM how to behave
    system_prompt = """You are a Marvel Comics expert assistant with access to 
specific comic book pages. Your job is to answer questions about the comics 
using ONLY the dialogue, captions, and text found in the provided pages.

Rules:
- Base your answer ONLY on the provided page text
- Cite which pages your information comes from (e.g. "According to page 2...")
- If the answer cannot be found in the provided pages, say so clearly
- Keep answers concise and focused on what the comics actually show
- Note that OCR text from comic pages may contain errors or missing words"""

    # user message — query + retrieved context
    user_message = f"""Question: {query}

Here are the most relevant comic pages I found:
{context_str}

Please answer the question based on what these pages show."""

    # call Claude API
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    return response.content[0].text


# ── Full RAG pipeline ──────────────────────────────────────────────────────────
def rag_pipeline(query: str, top_k: int = 5) -> dict:
    """
    Full RAG pipeline: retrieve → augment → generate.

    Returns:
        {
            "query":    original question,
            "sources":  list of retrieved page filenames,
            "context":  extracted OCR text per page,
            "answer":   Claude's generated answer
        }
    """
    print(f'\nRAG Query: "{query}"')
    print("─" * 55)

    # step 1: retrieve
    print("Step 1: Retrieving relevant pages...")
    retrieved = retrieve(query, top_k=top_k)
    print(f"  Found: {retrieved}")

    # step 2: augment
    print("Step 2: Extracting page context...")
    context_pages = get_context(retrieved)
    pages_with_text = sum(1 for p in context_pages if p["has_text"])
    print(f"  {pages_with_text}/{len(context_pages)} pages have readable OCR text")

    # step 3: generate
    print("Step 3: Generating answer with Claude...")
    answer = generate_answer(query, context_pages)

    print("\nAnswer:")
    print("─" * 55)
    print(answer)
    print("─" * 55)

    return {
        "query":   query,
        "sources": retrieved,
        "context": context_pages,
        "answer":  answer,
    }


def rag_answer(query: str, top_k: int = 5) -> str:
    """Convenience function — just returns the answer string."""
    result = rag_pipeline(query, top_k=top_k)
    return result["answer"]


# ── RAG evaluation ─────────────────────────────────────────────────────────────
# Unlike retrieval evaluation (precision/MRR/NDCG),
# RAG evaluation checks answer quality.
# Options:
#   1. Manual — read the answers and judge yourself (simplest)
#   2. Automated — use another LLM to score the answer (LLM-as-judge)

def evaluate_rag(queries_with_expected: list[dict]) -> None:
    """
    Run a set of RAG queries and print answers for manual evaluation.

    Args:
        queries_with_expected: list of {query, expected_keywords} dicts
            expected_keywords: list of words the answer should contain

    Example:
        evaluate_rag([
            {
                "query": "What does Punisher say before attacking?",
                "expected_keywords": ["punisher", "castle", "kill", "war"]
            }
        ])
    """
    print("\n" + "=" * 65)
    print("RAG EVALUATION")
    print("=" * 65)

    hits = 0
    for item in queries_with_expected:
        result = rag_pipeline(item["query"], top_k=5)
        answer_lower = result["answer"].lower()

        # simple keyword check — does the answer mention expected terms?
        keywords = item.get("expected_keywords", [])
        matched  = [kw for kw in keywords if kw.lower() in answer_lower]
        hit      = len(matched) >= len(keywords) // 2  # at least half must match

        hits += int(hit)
        status = "HIT" if hit else "MISS"
        print(f"\n[{status}] {item['query']}")
        print(f"  Keywords matched: {matched}/{keywords}")
        print(f"  Sources: {result['sources'][:3]}")

    print(f"\nRAG keyword hit rate: {hits}/{len(queries_with_expected)}")
    print("Note: Also read answers manually — keyword matching is a weak proxy")


# ── Demo queries ───────────────────────────────────────────────────────────────
RAG_DEMO_QUERIES = [
    "What does the Punisher say when confronting criminals?",
    "How does Daredevil describe his relationship with the law?",
    "What happens when Spider-Man meets his enemies?",
    "Does the Punisher show mercy to his enemies?",
    "What is Wilson Fisk's plan in the Daredevil comics?",
]


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # run a single demo query
    result = rag_pipeline(
        "What does the Punisher say when he confronts Wilson Fisk?",
        top_k=5
    )