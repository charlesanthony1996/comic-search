# Comic Search using NLP + CLIP

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![NLP](https://img.shields.io/badge/NLP-Comic%20Search-green)
![CLIP](https://img.shields.io/badge/OpenCLIP-ViT--B32-orange)

A hybrid comic search system that enables semantic and text-based retrieval from comic datasets. The project extracts comic pages from `.cbz` or `.pdf` files, builds image embeddings using CLIP, performs OCR on pages, and supports both visual-semantic and BM25 text retrieval.

---

## Features

- Extract pages from comic archives (`.cbz` preferred, `.pdf` supported as fallback)
- Automatic superhero/character inference from filenames and parent folder structure
- OCR text extraction using Tesseract
- Semantic image-text retrieval using OpenCLIP (ViT-B/32)
- BM25 keyword search over extracted OCR text
- Golden test set evaluation with Top-3 hit checking
- Visual display of ranked search results via Matplotlib

---

## Project Pipeline

```
Comic Files (.cbz / .pdf)
        ↓
  Page Extraction
        ↓
 Image Dataset Generation  (dataset/)
        ↓
  OCR Text Extraction
        ↓
  dataset_text.json
        ↓
CLIP Embedding Generation
        ↓
   clip_index.npz
        ↓
       Search
  ┌─────────────────────────┐
  │ CLIP Semantic Search    │
  │ BM25 Text Search        │
  └─────────────────────────┘
```

---

## Directory Structure

```
comic-search/
│
├── datasets/              # Input comic datasets (.cbz / .pdf) — git ignored
├── dataset/               # Extracted page images — git ignored
├── dataset_text.json      # OCR text corpus (generated)
├── clip_index.npz         # CLIP embedding index (generated) — git ignored
├── main.py                # Main script (all logic lives here)
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd comic-search
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR (system dependency)

- **Ubuntu/Debian:** `sudo apt install tesseract-ocr`
- **macOS:** `brew install tesseract`
- **Windows:** Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)

---

## Requirements

```
torch
numpy
pytesseract
open_clip
matplotlib
rank-bm25
```

> **Note:** `fitz` and `pypdf` are imported in the code but currently commented out in `requirements.txt`.
> Install `PyMuPDF` (which provides `fitz`) if you plan to use PDF extraction:
> ```
> pip install PyMuPDF
> ```

---

## Usage

> **Note:** The CLI argument parser (`argparse`) is currently commented out in `main.py`.
> To run the project, call the functions directly in `main.py` or uncomment the CLI block at the bottom.

### Step 1 — Extract comic pages

```python
extract_all()
```

Scans `datasets/` for `.cbz` and `.pdf` files, extracts pages as `.jpg` images into `dataset/`.
- `.cbz` is the preferred format
- `.pdf` is supported as a fallback via PyMuPDF (`fitz`)
- First and last 3 pages are skipped (covers/credits)

### Step 2 — Build CLIP image index

```python
build_index()
```

Generates CLIP (ViT-B/32) embedding vectors for all images in `dataset/` and saves them to `clip_index.npz`.

### Step 3 — Run semantic search

```python
run_search("punisher fighting with wilson fisk", top_k=5, show=True)
```

The `show=True` flag displays ranked results as images using Matplotlib.

**Example queries from the code:**

```python
run_search("punisher fighting with wilson fisk", top_k=5, show=True)
run_search("punisher and daredevil fighting", top_k=5, show=False)
run_search("the punisher thinking about his family and feeling sad", top_k=5, show=True)
run_search("microchip working on computers or hacking", top_k=5, show=True)
```

### Step 4 — Build OCR text corpus

```python
build_text_corpus()
```

Runs Tesseract OCR on every image in `dataset/` and saves extracted text to `dataset_text.json`.

### Step 5 — BM25 keyword search

```python
results = bm25_search("punisher fighting with wilson fisk", top_k=5)
print(results)
```

Returns a list of `(score, filename)` tuples ranked by BM25 relevance over the OCR corpus.

### Step 6 — Run evaluation

```python
evaluate()
```

Runs the golden test set against the CLIP index. Reports Top-3 hit/miss for each query and prints a final score.

---

## Character Map

The project uses a keyword-based character map to infer superhero identity from filenames and folder names:

| Keyword | Character |
|---|---|
| `venom` | venom |
| `wolverine`, `logan` | wolverine |
| `amazing`, `spider` | spiderman |
| `punisher`, `frank`, `castle` | punisher |
| `matt`, `murdock`, `daredevil`, `devil` | daredevil |

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core language |
| OpenCLIP (ViT-B/32) | Semantic image-text embeddings |
| Tesseract OCR | Text extraction from comic pages |
| BM25 (`rank-bm25`) | Keyword-based text retrieval |
| PyMuPDF (`fitz`) | PDF page extraction |
| `zipfile` (stdlib) | `.cbz` archive extraction |
| NumPy | Embedding storage and cosine similarity |
| Matplotlib | Visual display of search results |
| Pillow | Image loading and preprocessing |

---

## Evaluation

The project includes a hardcoded golden test set covering 7 queries across 3 characters:

| Character | Queries |
|---|---|
| Spider-Man | 2 |
| Venom | 2 |
| Wolverine | 3 |

Punisher and Daredevil mixed-appearance queries are noted as a TODO in the code. Each query is evaluated for a Top-3 hit — whether the expected character appears in any of the top 3 retrieved filenames.

---

## Known Limitations

- The CLI (`argparse`) block is currently commented out — functions must be called directly in `main.py`
- Character inference falls back to `None` if the folder name is unrecognized and no keyword matches the filename (visible in `dataset_text.json` as `None_page_...`)
- `pypdf` is imported but commented out in `requirements.txt`
- The Dockerfile is currently empty

---

## Notes — Git Ignored Files

These files are excluded via `.gitignore` and must be generated locally:

```
datasets/       # place your .cbz / .pdf comics here
dataset/        # auto-generated by extract_all()
clip_index.npz  # auto-generated by build_index()
.env
```

---

## Future Improvements

- Re-enable and finalize the CLI (`argparse`) interface
- Fix `None_` prefix in character inference for unrecognized folders
- Hybrid score fusion between CLIP + BM25 results
- FAISS vector indexing for faster retrieval at scale
- Web UI for interactive search
- Improved OCR preprocessing (denoising, deskewing)
- Complete the Dockerfile for containerized deployment

---

## Contributors

- Charles
- Dayas
- Rustam