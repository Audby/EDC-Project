"""Generate report figures for MNIST task 1 (full-template NN)."""
import time
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def nn_predict_chunked(X_test, X_train, y_train, chunk=1000):
    train_sq = np.sum(X_train ** 2, axis=1)
    preds = np.empty(X_test.shape[0], dtype=y_train.dtype)
    for start in range(0, X_test.shape[0], chunk):
        end = start + chunk
        Xc = X_test[start:end]
        test_sq = np.sum(Xc ** 2, axis=1, keepdims=True)
        dists = test_sq + train_sq - 2.0 * Xc @ X_train.T
        nn_idx = np.argmin(dists, axis=1)
        preds[start:end] = y_train[nn_idx]
    return preds


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


data = loadmat("MNIST files/data_all.mat")
X_train = data["trainv"].astype(np.float32)
X_test = data["testv"].astype(np.float32)
y_train = data["trainlab"].ravel().astype(np.int64)
y_test = data["testlab"].ravel().astype(np.int64)

t0 = time.time()
y_pred = nn_predict_chunked(X_test, X_train, y_train, chunk=1000)
elapsed = time.time() - t0
print(f"Classification took {elapsed:.2f} s")

cm = confusion_matrix(y_test, y_pred, 10)
err = np.mean(y_test != y_pred)
print(f"Error rate: {err*100:.2f}% ({int(err*len(y_test))} of {len(y_test)})")

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
ax.set_title("Full-template NN (60 000 templates)")
for i in range(10):
    for j in range(10):
        v = cm[i, j]
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() * 0.5 else "black",
                fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig("Images/ConfusionMatrices/mnist_full_nn_confusion.png", dpi=150)
plt.close(fig)

mis = np.where(y_test != y_pred)[0][:10]
fig, axes = plt.subplots(5, 2, figsize=(5, 11))
fig.suptitle("Misclassified Images", fontsize=14)
for ax, idx in zip(axes.ravel(), mis):
    ax.imshow(X_test[idx].reshape(28, 28), cmap="gray")
    ax.set_title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}", fontsize=11)
    ax.axis("off")
fig.tight_layout()
fig.savefig("Images/MNIST/mnist_wrong.png", dpi=150, bbox_inches="tight")
plt.close(fig)

cor = np.where(y_test == y_pred)[0][:10]
fig, axes = plt.subplots(5, 2, figsize=(5, 11))
fig.suptitle("Correctly Classified Images", fontsize=14)
for ax, idx in zip(axes.ravel(), cor):
    ax.imshow(X_test[idx].reshape(28, 28), cmap="gray")
    ax.set_title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}", fontsize=11)
    ax.axis("off")
fig.tight_layout()
fig.savefig("Images/MNIST/mnist_correct.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("Saved figures to Images/")

off = cm.copy()
np.fill_diagonal(off, 0)
flat = [(i, j, off[i, j]) for i in range(10) for j in range(10) if off[i, j] > 0]
flat.sort(key=lambda x: -x[2])
print("Top confusions (true -> pred : count):")
for i, j, v in flat[:8]:
    print(f"  {i} -> {j} : {v}")
