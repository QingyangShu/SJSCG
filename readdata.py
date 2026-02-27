import scipy.io
import numpy as np

def load_multiview_data_with_mask(mat_path):
    mat = scipy.io.loadmat(mat_path)
    X_cell = mat['X']
    y = mat['y'].squeeze()
    M_mask = mat['M_mask']  

    X_views = [X_cell[i, 0] for i in range(X_cell.shape[0])]
    cluster_number = len(np.unique(y))

    return X_views, y, cluster_number, M_mask
