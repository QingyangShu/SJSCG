import scipy.io
import numpy as np

def load_multiview_data(mat_path, missing_rate=0.0, seed=42):
    mat = scipy.io.loadmat(mat_path)
    X_cell = mat['X']
    y = mat['y'].squeeze()

    X_views = [X_cell[i, 0] for i in range(X_cell.shape[0])]
    n_views = len(X_views)
    n_samples = X_views[0].shape[0]

    rng = np.random.default_rng(seed)
    M_mask = np.ones((n_samples, n_views), dtype=np.uint8)

    missing_per_view = int(n_samples * missing_rate)
    for v in range(n_views):
        indices = rng.permutation(n_samples)[:missing_per_view]
        M_mask[indices, v] = 0

    cluster_number = len(np.unique(y))

    return X_views, y, cluster_number, M_mask
