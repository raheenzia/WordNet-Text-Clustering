import numpy as np
from collections import Counter
from sklearn.preprocessing import normalize


def purity(clusters, true_labels):
    total = sum(len(c) for c in clusters)
    if total == 0:
        return 0.0
    score = 0
    for c in clusters:
        if not c:
            continue
        counts = Counter(true_labels[i] for i in c)
        score += max(counts.values())
    return score / total


def entropy(clusters, true_labels):
    total = sum(len(c) for c in clusters)
    if total == 0:
        return 0.0
    result = 0.0
    for c in clusters:
        if not c:
            continue
        size = len(c)
        counts = Counter(true_labels[i] for i in c)
        ice= 0.0
        for count in counts.values():
            p = count / size
            if p > 0:
                ice -= p * np.log2(p)
        result += (size / total) * ice
    return result


def overall_similarity(clusters, X):
    X_norm = normalize(X, norm='l2')
    total  = sum(len(c) for c in clusters)
    if total == 0:
        return 0.0
    score = 0.0
    for c in clusters:
        if not c:
            continue
        centroid = np.asarray(X_norm[c].mean(axis=0)).flatten()
        score   += (len(c) / total) * float(np.dot(centroid, centroid))
    return score
