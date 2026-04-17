import time
import numpy as np
from scipy.io import loadmat


def sq_euclid_dist(x, z):
    return np.sum((x-z)**2, axis=-1)

def nearest_neighbor(X_test, X_train, y_train):
    distances = sq_euclid_dist(X_test, X_train)
    nn_index = np.argmin(distances)
    return y_train[nn_index]

data = loadmat("MNIST files/data_all.mat")
num_train = int(data["num_train"].squeeze())
num_test = int(data["num_test"].squeeze())
vec_size = int(data["vec_size"].squeeze())
row_size = int(np.sqrt(vec_size))
col_size = int(np.sqrt(vec_size))

trainv = data["trainv"].astype(np.float32)
testv = data["testv"].astype(np.float32)
trainlab = data["trainlab"].ravel()
testlab = data["testlab"].ravel()


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
        print(f" chunk {end}/{X_test.shape[0]} done")

    return preds

def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def error_rate(y_true, y_pred):
    return np.mean(y_true != y_pred)

NUM_CLASSES = 10

X_train = trainv.astype(np.float32)
X_test = testv.astype(np.float32)
y_train = trainlab.ravel().astype(np.int64)
y_test = testlab.ravel().astype(np.int64)

t0 = time.time()
y_pred = nn_predict_chunked(X_test, X_train, y_train, chunk=1000)
print(f"Classification took {time.time() - t0:1f} s")

cm = confusion_matrix(y_test, y_pred, NUM_CLASSES)
err = error_rate(y_test, y_pred)

print("\nConfusion matrix (rows = true class, columns = predictions):")
print(cm)
print(f"\nError rate: {err * 100:.2f}% ({int(err * len(y_test))} of {len(y_test)} errors)")