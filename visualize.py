from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from sklearn.feature_selection import f_classif
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.svm import LinearSVC
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


def scatter_features(
    X: np.ndarray,
    y: np.ndarray,
    out_path: str,
    title: str = "PBP Features - True Targets",
    label_names: Optional[dict] = None,
    figsize=(7, 6),
    plot_separators: bool = False,
    show_fig: bool = False,
    feature_names: Optional[list] = None,
) -> None:
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

    # Colors for separators distinct from sample colors
    sep_colors = [
        'cyan', 'magenta', 'lime', 'turquoise', 'indigo', 'gold',
        'slateblue', 'darkcyan', 'deeppink', 'olive'
    ]

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
        xlbl = feature_names[best3[0]] if feature_names and len(feature_names) > best3[0] else f"Feature {best3[0] + 1}"
        ylbl = feature_names[best3[1]] if feature_names and len(feature_names) > best3[1] else f"Feature {best3[1] + 1}"
        zlbl = feature_names[best3[2]] if feature_names and len(feature_names) > best3[2] else f"Feature {best3[2] + 1}"
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)
        ax.set_zlabel(zlbl)

        # Optional: plot linear separator planes
        if plot_separators:
            try:
                X3 = X[:, best3]
                uniq = sorted(set(int(v) for v in y))
                sep_handles = []
                sep_labels = []
                if len(uniq) == 2:
                    clf = LinearSVC(dual=False, max_iter=5000, random_state=0)
                    clf.fit(X3, y)
                    w = clf.coef_[0]
                    b = clf.intercept_[0]
                    # Create grid over x-y, solve for z
                    xlim = (np.min(X3[:, 0]), np.max(X3[:, 0]))
                    ylim = (np.min(X3[:, 1]), np.max(X3[:, 1]))
                    xx, yy = np.meshgrid(
                        np.linspace(xlim[0], xlim[1], 20),
                        np.linspace(ylim[0], ylim[1], 20),
                    )
                    if abs(w[2]) > 1e-12:
                        zz = (-b - w[0] * xx - w[1] * yy) / w[2]
                        color_sep = sep_colors[0]
                        ax.plot_surface(xx, yy, zz, alpha=0.15, color=color_sep, edgecolor='none')
                        # Equation label: w0 x + w1 y + w2 z + b = 0
                        eq = f"{w[0]:.2f}x + {w[1]:.2f}y + {w[2]:.2f}z + {b:.2f} = 0"
                        sep_handles.append(Patch(facecolor=color_sep, edgecolor='none', alpha=0.6))
                        sep_labels.append(f"separator: {eq}")
                else:
                    # One-vs-rest planes (may clutter)
                    clf = LinearSVC(dual=False, max_iter=5000, random_state=0)
                    for i_cls, one in enumerate(uniq):
                        yy_bin = (y == one).astype(int)
                        clf.fit(X3, yy_bin)
                        w = clf.coef_[0]
                        b = clf.intercept_[0]
                        xlim = (np.min(X3[:, 0]), np.max(X3[:, 0]))
                        ylim = (np.min(X3[:, 1]), np.max(X3[:, 1]))
                        xx, yy_grid = np.meshgrid(
                            np.linspace(xlim[0], xlim[1], 15),
                            np.linspace(ylim[0], ylim[1], 15),
                        )
                        if abs(w[2]) > 1e-12:
                            zz = (-b - w[0] * xx - w[1] * yy_grid) / w[2]
                            color_sep = sep_colors[i_cls % len(sep_colors)]
                            ax.plot_surface(xx, yy_grid, zz, alpha=0.10, color=color_sep, edgecolor='none')
                            eq = f"{w[0]:.2f}x + {w[1]:.2f}y + {w[2]:.2f}z + {b:.2f} = 0"
                            cls_name = label_names.get(one, str(one)) if label_names else str(one)
                            sep_handles.append(Patch(facecolor=color_sep, edgecolor='none', alpha=0.6))
                            sep_labels.append(f"sep[{cls_name}]: {eq}")
                # Adjust axes limits with small margin
                x_min, x_max = np.min(X3[:, 0]), np.max(X3[:, 0])
                y_min, y_max = np.min(X3[:, 1]), np.max(X3[:, 1])
                z_min, z_max = np.min(X3[:, 2]), np.max(X3[:, 2])
                mx = 0.05 * (x_max - x_min + 1e-6)
                my = 0.05 * (y_max - y_min + 1e-6)
                mz = 0.05 * (z_max - z_min + 1e-6)
                ax.set_xlim(x_min - mx, x_max + mx)
                ax.set_ylim(y_min - my, y_max + my)
                ax.set_zlim(z_min - mz, z_max + mz)

                # Merge legend entries with separators
                h, l = ax.get_legend_handles_labels()
                if sep_handles:
                    ax.legend(h + sep_handles, l + sep_labels, loc="best", fontsize=8)
            except Exception:
                pass
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
        xlbl = feature_names[best2[0]] if feature_names and len(feature_names) > best2[0] else f"Feature {best2[0] + 1}"
        ylbl = feature_names[best2[1]] if feature_names and len(feature_names) > best2[1] else f"Feature {best2[1] + 1}"
        plt.xlabel(xlbl)
        plt.ylabel(ylbl)

        # Optional: plot linear separator lines
        if plot_separators:
            try:
                X2 = X[:, best2]
                uniq = sorted(set(int(v) for v in y))
                sep_handles = []
                sep_labels = []
                if len(uniq) == 2:
                    clf = LinearSVC(dual=False, max_iter=5000, random_state=0)
                    clf.fit(X2, y)
                    w = clf.coef_[0]
                    b = clf.intercept_[0]
                    xs = np.linspace(np.min(X2[:, 0]), np.max(X2[:, 0]), 100)
                    if abs(w[1]) > 1e-12:
                        ys = (-(w[0] * xs + b) / w[1])
                        color_sep = sep_colors[0]
                        line, = plt.plot(xs, ys, color=color_sep, linewidth=1.5, alpha=0.9)
                        eq = f"{w[0]:.2f}x + {w[1]:.2f}y + {b:.2f} = 0"
                        sep_handles.append(line)
                        sep_labels.append(f"separator: {eq}")
                else:
                    clf = LinearSVC(dual=False, max_iter=5000, random_state=0)
                    xs = np.linspace(np.min(X[:, best2[0]]), np.max(X[:, best2[0]]), 100)
                    for i_cls, one in enumerate(uniq):
                        yy_bin = (y == one).astype(int)
                        clf.fit(X2, yy_bin)
                        w = clf.coef_[0]
                        b = clf.intercept_[0]
                        if abs(w[1]) > 1e-12:
                            ys = (-(w[0] * xs + b) / w[1])
                            color_sep = sep_colors[i_cls % len(sep_colors)]
                            line, = plt.plot(xs, ys, color=color_sep, linewidth=1.2, alpha=0.8)
                            eq = f"{w[0]:.2f}x + {w[1]:.2f}y + {b:.2f} = 0"
                            cls_name = label_names.get(one, str(one)) if label_names else str(one)
                            sep_handles.append(line)
                            sep_labels.append(f"sep[{cls_name}]: {eq}")

                # Axis limits with margin
                x_min, x_max = np.min(X2[:, 0]), np.max(X2[:, 0])
                y_min, y_max = np.min(X2[:, 1]), np.max(X2[:, 1])
                mx = 0.05 * (x_max - x_min + 1e-6)
                my = 0.05 * (y_max - y_min + 1e-6)
                plt.xlim(x_min - mx, x_max + mx)
                plt.ylim(y_min - my, y_max + my)

                # Merge legend entries with separators
                h, l = plt.gca().get_legend_handles_labels()
                if sep_handles:
                    plt.legend(h + sep_handles, l + sep_labels, loc="best", fontsize=8)
            except Exception:
                pass
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

    # plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show_fig:
        plt.show()
    plt.close()