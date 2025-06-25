import json
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def reduce_dimensions(embeddings, n_components=2):
    """Reduces the dimensionality of the embeddings using PCA."""
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)

    # """Reduces the dimensionality of the embeddings using t-SNE."""
    # tsne = TSNE(n_components=n_components, perplexity=1)
    # reduced_embeddings = tsne.fit_transform(embeddings)

    return reduced_embeddings