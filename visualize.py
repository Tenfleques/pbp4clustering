from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from sklearn.feature_selection import f_classif
import logging
logging.getLogger('matplotlib.font_manager').disabled = True


def scatter_pca_2d(X: np.ndarray, y: np.ndarray, out_path: str, title: str = "PBP (PCA) - True Targets",
                    label_names: Optional[dict] = None, figsize=(7, 6)) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    pca = PCA(n_components=2, random_state=0)
    X2 = pca.fit_transform(X)

    plt.figure(figsize=figsize)
    scatter = plt.scatter(X2[:, 0], X2[:, 1], c=y, cmap="tab10", s=1, alpha=0.8)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    if label_names:
        # Build legend
        handles, _ = scatter.legend_elements()
        labels = []
        for k in sorted(set(int(v) for v in y)):
            labels.append(label_names.get(k, str(k)))
        plt.legend(handles, labels, title="Class", loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def scatter_features(X: np.ndarray, y: np.ndarray, out_path: str,
                     title: str = "PBP Features - True Targets",
                     label_names: Optional[dict] = None, figsize=(7, 6)) -> None:
    """
    Visualize PBP feature vectors directly (no PCA), using 1D/2D/3D scatter depending on dimensionality.

    - If X has >=3 columns: plots first three features in 3D
    - If X has >=2 columns: plots first two features
    - If X has 1 column: plots feature vs sample index
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    num_features = X.shape[1]
    unique_labels = sorted(set(int(v) for v in y))

    colors = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'brown', 'pink', 'gray', 'black',
              'darkblue', 'darkred', 'darkgreen', 'olive', 'indigo', 'coral', 'maroon', 'hotpink', 'darkgray', 'dimgray',
              'lightblue', 'salmon', 'lime', 'gold', 'violet', 'peachpuff', 'sienna', 'lightpink', 'silver', 'slategray',
              'navy', 'crimson', 'forestgreen', 'khaki', 'plum', 'sandybrown', 'saddlebrown', 'deeppink', 'lightgray', 'teal',
              'royalblue', 'firebrick', 'seagreen', 'goldenrod', 'magenta', 'chocolate', 'rosybrown', 'palevioletred', 'steelblue']
    markers = ['o', 's', 'd', 'v', '^', 'p', 'h', 'x', 'D', '1', '2', '3', '4', '8', 'P', '*', 'H', '+', 'X', '_',
               '|', ',', '.', '<', '>', 'p', 'h', 'x', 'D', '1', '2', '3', '4', '8', 'P', '*', 'H', '+', 'X', '_',
               '|', ',', '.', '<', '>', 'p', 'h', 'x', 'D']

    plt.figure(figsize=figsize)

    if num_features >= 3:
        # Select top-3 features by ANOVA F-score for best linear separability
        try:
            f_scores, _ = f_classif(X, y)
            best3 = np.argsort(np.nan_to_num(f_scores, nan=-np.inf))[::-1][:3]
        except Exception:
            best3 = np.array([0, 1, 2])

        ax = plt.axes(projection='3d')
        for idx, cls in enumerate(unique_labels):
            mask = (y == cls)
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            ax.scatter(X[mask, best3[0]], X[mask, best3[1]], X[mask, best3[2]], s=1, c=color,
                       label=label_names.get(cls, str(cls)) if label_names else str(cls), marker=marker, alpha=0.85)
        ax.set_xlabel(f"Feature {best3[0] + 1}")
        ax.set_ylabel(f"Feature {best3[1] + 1}")
        ax.set_zlabel(f"Feature {best3[2] + 1}")
    elif num_features >= 2:
        # Select top-2 features by ANOVA F-score
        try:
            f_scores, _ = f_classif(X, y)
            best2 = np.argsort(np.nan_to_num(f_scores, nan=-np.inf))[::-1][:2]
        except Exception:
            best2 = np.array([0, 1])

        for idx, cls in enumerate(unique_labels):
            mask = (y == cls)
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            plt.scatter(X[mask, best2[0]], X[mask, best2[1]], s=1, c=color,
                        label=label_names.get(cls, str(cls)) if label_names else str(cls), marker=marker, alpha=0.85)
        plt.xlabel(f"Feature {best2[0] + 1}")
        plt.ylabel(f"Feature {best2[1] + 1}")
    else:
        # 1D: plot feature vs sample index
        start = 0
        for idx, cls in enumerate(unique_labels):
            mask = (y == cls)
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            idxs = np.where(mask)[0]
            plt.scatter(idxs, X[mask, 0], s=1, c=color,
                        label=label_names.get(cls, str(cls)) if label_names else str(cls), marker=marker, alpha=0.85)
        plt.xlabel("Sample Index")
        plt.ylabel("Feature 1")

    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    plt.close()