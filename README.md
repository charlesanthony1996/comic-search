# Comic Search using NLP + CLIP

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![NLP](https://img.shields.io/badge/NLP-Comic%20Search-green)
![CLIP](https://img.shields.io/badge/OpenCLIP-ViT--B32-orange)

A hybrid comic search system that enables semantic and text-based retrieval from comic datasets. The project extracts comic pages from `.cbz` or `.pdf` files, builds image embeddings using CLIP, performs OCR on pages, and supports both visual-semantic and BM25 text retrieval.

---

## Features

- Extract pages from comic archives (`.cbz` preferred, `.pdf` supported)
- Automatic superhero/character inference from filenames and folder structure
- OCR text extraction using Tesseract
- Semantic image-text retrieval using OpenCLIP (CLIP)
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
 Image Dataset Generation
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
├── datasets/              # Input comic datasets (.cbz / .pdf)
├── dataset/               # Extracted page images
├── dataset_text.json      # OCR text corpus
├── clip_index.npz         # Generated CLIP embedding index
├── main.py                # Main entry point
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
torch>=2.0
numpy>=1.26
pytesseract>=0.3.10
PyMuPDF>=1.25
pypdf>=5.0
open-clip-torch>=2.24
Pillow>=10.0
matplotlib>=3.8
rank-bm25>=0.2
```

---

## Usage

### Extract comic pages

```bash
python main.py --extract
```

Extracts pages from `.cbz` and `.pdf` files in the `datasets/` folder into the `dataset/` directory. `.cbz` format is preferred; `.pdf` is supported as a fallback.

### Build CLIP image index

```bash
python main.py --index
```

Generates CLIP embedding vectors for all extracted pages and saves them to `clip_index.npz`.

### Run semantic search

```bash
python main.py --search "punisher fighting with wilson fisk" --show
```

The `--show` flag displays the top results visually using Matplotlib.

**Example queries:**

```bash
python main.py --search "spider-man swinging"
python main.py --search "punisher fighting with daredevil"
python main.py --search "black symbiote monster"
```

### BM25 text search (OCR-based)

Build the OCR text corpus:

```python
build_text_corpus()
```

Run a BM25 keyword search:

```python
bm25_search("punisher fighting with wilson fisk")
```

### Run evaluation

```bash
python main.py --eval
```

Evaluates retrieval quality against the golden test set using Top-3 hit checking and character-based relevance validation.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core language |
| OpenCLIP | Semantic image-text embeddings |
| Tesseract OCR | Text extraction from comic pages |
| BM25 (rank-bm25) | Keyword-based text retrieval |
| PyMuPDF | PDF page extraction |
| NumPy | Embedding storage and similarity |
| Matplotlib | Visual display of search results |
| Pillow | Image processing |

---

## Evaluation

The project includes a golden test set for retrieval quality measurement.

Current evaluation metrics:
- **Top-3 hit rate** — checks if the correct page appears in the top 3 results
- **Character-based relevance** — validates results based on superhero/character identity inferred from filenames

---

## Notes

Large datasets and generated index files are excluded via `.gitignore` to keep the repository lightweight:

```
datasets/
dataset/
clip_index.npz
.env
```

These files must be generated locally after cloning using the `--extract` and `--index` commands.

---

## Future Improvements

- Hybrid score fusion between CLIP + BM25 results
- FAISS vector indexing for faster retrieval at scale
- Web UI for interactive search
- Improved OCR preprocessing (denoising, deskewing)
- Multimodal query support (image + text)

---

## Contributors

- Charles
- Dayas
- Rustam