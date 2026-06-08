from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def bisecting_kmeans(X, k, n_trials=3):
    X_norm = normalize(X, norm='l2')  # unit vecs >>  dot prod == cosine sim

    clusters = [list(range(X_norm.shape[0]))]

    while len(clusters) < k:
        largest_idx = max(range(len(clusters)), key=lambda i: len(clusters[i])) #largest cluster
        to_split    = clusters.pop(largest_idx)

        if len(to_split) < 2:        
            clusters.append(to_split)
            break

        subset = X_norm[to_split]

        best_labels, best_sse = None, float('inf')

        for _ in range(n_trials):
            km = KMeans(n_clusters=2, n_init=1, max_iter=100)
            labels = km.fit_predict(subset)
            if km.inertia_ < best_sse:
                best_sse, best_labels = km.inertia_, labels

        part_a = [to_split[i] for i, l in enumerate(best_labels) if l == 0]
        part_b = [to_split[i] for i, l in enumerate(best_labels) if l == 1]

        if part_a: clusters.append(part_a)
        if part_b: clusters.append(part_b)

    return clusters
