from main import run_search, bm25_search, rrf_search, evaluate_all, compare, plot_comparison

from rag import rag_pipeline

print("demo 1 - clip semantic search")
run_search("punisher fighting with wilson fisk", top_k = 5, show=True)


# this demo is for dialogue matching
print("demo 2 - bm25 text search")
results = bm25_search("one batch two batch penny and dime", top_k=5)

for score, fname in results:
    print(f"{score:.4f} {fname}")

print("demo 3 - rrf fusion")
run_search_rrf = lambda q, top_k: rrf_search(q, top_k=top_k)

# show this visually
from main import image_dir
from PIL import Image
import matplotlib.pyplot as plt

query = "daredevil fighting at night"
results = rrf_search(query, top_k=5)
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle(f"rrf: '{query}'", fontsize=12)

for ax, fname in zip(axes, results):

    img = Image.open(image_dir / fname)
    
    ax.imshow(img)

    ax.axis("off")

    ax.set_title(fname[:25], fontsize=6)

plt.tight_layout()

plt.show()

print("demo 4 comparison")

clip_results = evaluate_all(run_search, k = 5, mode="clip")
bm25_results = evaluate_all(bm25_search, k = 5, mode="bm25")
rrf_results = evaluate_all(rrf_search, k = 5, mode="rrf")
compare(clip_results, rrf_results)
plot_comparison(clip_results, bm25_results, rrf_results)

print("demo 5 rag:- question answering from metrics")
rag_pipeline("what does the punisher say when confronting wilson fisk?" ,top_k =5)
