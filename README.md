# WordNet-Based Text Document Clustering on 20 Newsgroups

A Python implementation of **Sedding & Kazakov (2013)** paper: *"WordNet-based Text Document Clustering"* — replicated on the **20 Newsgroups** dataset.

## Paper Summary (3 lines)

The paper tests whether adding **WordNet synonyms and hypernyms** (with PoS tagging) improves text clustering. The key finding: **it does NOT help** — background knowledge adds too much noise (wrong word senses), making clusters worse than simple bag-of-words. The best results came from **Baseline** (no WordNet, no PoS).

## Corpus Choice (Difference from Paper)

| | **Paper (Reuters-21578)** | **This Implementation (20 Newsgroups)** |
|--|---------------------------|------------------------------------------|
| Documents | ~9,000–22,000 | 18,846 |
| Categories | ~50–90 (single-label) | **20 fixed categories** |
| Balance | Unbalanced (capped at 20/50/100) | **Perfectly balanced** (1,000 per category) |
| Domain | Financial news | Usenet posts (tech, sports, politics, etc.) |


## Experiment Steps

1. **Load** 20 Newsgroups (full dataset, no sub-sampling)
2. **Preprocess** per config:
   - PoS tagging (NLTK)
   - Stopword removal (keep N/V/Adj only)
   - Lemmatization (WordNet morphology)
   - WordNet enrichment (synset IDs + hypernyms)
   - Pruning (remove terms with df < threshold)
   - TF‑IDF weighting (paper's exact `log2` formula)
3. **Cluster** with bisecting k‑means (`k = 16, 32, 64`)
4. **Evaluate** using purity, entropy, overall similarity
5. **Average** 3 runs per configuration (for stability)

## Configurations (from Paper Table 2)

| Config | PoS Tags | Synonyms | Hypernyms |
|--------|----------|----------|-----------|
| **Baseline** | ❌ | ❌ | ❌ |
| **PoS_Only** | ✅ | ❌ | ❌ |
| **Syns** | ✅ | ✅ | ❌ |
| **Hyper_5** | ✅ | ✅ | 5 levels |
| **Hyper_All** | ✅ | ✅ | All levels |

## Cluster Sizes (k)

We tested **k = 15, 20, 24** to better match the 20 Newsgroups category count - — to test under, near, and over the true number of categories (20).
> NOTE: In the paper: **16, 32, 64** clusters 


## Results Summary

| Config | Purity (k=32) | Entropy (k=32) | Finding |
|--------|--------------|----------------|---------|
| **Baseline** | **0.266** | **3.20** | Best |
| PoS_Only | 0.256 | 3.26 | Slightly worse |
| Syns | 0.130 | 4.03 | Much worse |
| Hyper_5 | 0.140 | 3.93 | Worse |
| Hyper_All | 0.136 | 3.95 | Worst |

**Replicates paper's main conclusion:** WordNet enrichment **degrades** clustering quality due to polysemy noise.
