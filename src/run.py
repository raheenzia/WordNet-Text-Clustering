import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.datasets import fetch_20newsgroups

from config import EXPERIMENT_CONFIGS, CLUSTER_SIZES, NUM_RUNS, RANDOM_SEED, PRUNING_THRESHOLD
from src.preprocess import build_document_vectors
from src.clustering import bisecting_kmeans
from src.evaluation import purity, entropy, overall_similarity

def load_data():
    print("Loading 20 Newsgroups...")
    data = fetch_20newsgroups(
        subset='all',
        remove=('headers', 'footers', 'quotes')
    )
    print(f"  {len(data.data)} docs | {len(data.target_names)} categories")
    return data.data, list(data.target)

def run_experiments(docs, labels):
    rows = []

    for config_name, config in EXPERIMENT_CONFIGS.items():
        print(f"\n{'─'*30}")
        print(f"Config: {config_name}")
        print(f"{'─'*30}")

        X = build_document_vectors(docs, config, PRUNING_THRESHOLD)

        for k in CLUSTER_SIZES:
            run_purities, run_entropies, run_similarities = [], [], []

            for run in range(NUM_RUNS):
                np.random.seed(RANDOM_SEED + run)
                clusters = bisecting_kmeans(X, k)

                run_purities.append(purity(clusters, labels))
                run_entropies.append(entropy(clusters, labels))
                run_similarities.append(overall_similarity(clusters, X))

            p = float(np.mean(run_purities))
            e = float(np.mean(run_entropies))
            s = float(np.mean(run_similarities))

            print(f"  k={k:2d}: purity={p:.3f}  entropy={e:.3f}  similarity={s:.3f}")

            rows.append({
                'config': config_name,
                'k': k,
                'purity': round(p, 4),
                'entropy': round(e, 4),
                'similarity': round(s, 4),
            })

    return pd.DataFrame(rows)

def save_results(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, 'results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    pivot = df.pivot_table(
        index='config', columns='k',
        values=['purity', 'entropy', 'similarity']
    ).round(4)
    pivot_path = os.path.join(out_dir, 'results_pivot.csv')
    pivot.to_csv(pivot_path)
    print(f"Saved: {pivot_path}")

    return pivot

def plot_results(df, out_dir):
    config_order = list(EXPERIMENT_CONFIGS.keys())
    colors = {
        'purity': ('steelblue','Purity'),
        'entropy': ('firebrick', 'Entropy'),
        'similarity': ('seagreen','Similarity'),
    }

    x_pos, x_labels = [], []
    x = 0
    separators = []

    for k in CLUSTER_SIZES:
        for cfg in config_order:
            x_labels.append(f"{cfg}\nk={k}")
            x_pos.append(x)
            x += 1
        separators.append(x - 0.5)

    fig, ax = plt.subplots(figsize=(14, 5))

    for metric, (color, label) in colors.items():
        y = []
        for k in CLUSTER_SIZES:
            for cfg in config_order:
                row = df[(df['config'] == cfg) & (df['k'] == k)]
                y.append(float(row[metric].values[0]) if not row.empty else 0)
        ax.plot(x_pos, y, marker='o', label=label,
                color=color, linewidth=1.5, markersize=4)

    for sep in separators[:-1]:
        ax.axvline(x=sep, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=7, rotation=30, ha='right')
    # ax.set_ylim(0, 1.35)
    ax.set_ylabel('Score')
    ax.set_title ('Clustering Results — 20 Newsgroups')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    plt.tight_layout()
    path = os.path.join(out_dir, 'results.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

if __name__ == '__main__':
    # out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    out_dir = "results"

    docs, labels = load_data()
    df = run_experiments(docs, labels)
    pivot = save_results(df, out_dir)
    plot_results(df, out_dir)

    print("\n===== Summary =================================")
    print(pivot.to_string())
    print("\nDone. Results in results/")
