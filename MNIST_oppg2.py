import time
import numpy as np
from scipy.io import loadmat
from sklearn.cluster import KMeans

#Loading the MNIST dataset
data = loadmat("MNIST files/data_all.mat")

num_train = int(data["num_train"].squeeze())
num_test = int(data["num_test"].squeeze())
vec_size = int(data["vec_size"].squeeze())
row_size = int(np.sqrt(vec_size))
col_size = int(np.sqrt(vec_size))

trainv = data["trainv"].astype(np.float32)
testv = data["testv"].astype(np.float32)
trainlab = data["trainlab"].ravel().astype(np.int64)
testlab = data["testlab"].ravel().astype(np.int64)

print("trainv shape:", trainv.shape)
print("testv shape:", testv.shape)
print("trainlab shape:", trainlab.shape)
print("testlab shape:", testlab.shape)
print("vec_size:", vec_size)
print("image size:", row_size, "x", col_size)


#Checking how many training samples there are per class
for digit in range(10):
    count = np.sum(trainlab == digit)
    print(f"Digit {digit}: {count} training samples")


# ------------------------------------
# Task 2a: clustering each class into M = 64 templates
# ------------------------------------
M = 64
NUM_CLASSES = 10

cluster_centers_list = []
cluster_labels_list = []

t0 = time.time()

for digit in range(NUM_CLASSES):
    # Select only training vectors from this digit class
    train_digit = trainv[trainlab == digit]

    print(f"\nClustering digit {digit}...")
    print("Class data shape:", train_digit.shape)

    # Performs clustering to make M cluster centers (templates) for this digit class
    kmeans = KMeans(n_clusters=M, random_state=42, n_init=10)
    idx_i = kmeans.fit_predict(train_digit)
    C_i = kmeans.cluster_centers_.astype(np.float32)

    print(f"Digit {digit}: cluster center matrix shape = {C_i.shape}")

    #Builds final template set for all digits together 
    cluster_centers_list.append(C_i)
    cluster_labels_list.append(np.full(M, digit, dtype=np.int64))

clustering_time = time.time() - t0

# Combine all classes into one template set
cluster_templates = np.vstack(cluster_centers_list)     # shape: (640, 784)
cluster_template_labels = np.concatenate(cluster_labels_list)  # shape: (640,)

print("\n--- Task 2a finished ---")
print("cluster_templates shape:", cluster_templates.shape)
print("cluster_template_labels shape:", cluster_template_labels.shape)
print(f"Clustering took {clustering_time:.2f} seconds")

#classifies test images using NN, but against clustered templates instead of all training samples
#classifies test samples in chunks to save memory 
def nn_predict_chunked(X_test, X_templates, y_templates, chunk=1000):
    template_sq = np.sum(X_templates ** 2, axis=1)
    preds = np.empty(X_test.shape[0], dtype=y_templates.dtype)

    for start in range(0, X_test.shape[0], chunk):
        end = min(start + chunk, X_test.shape[0])
        Xc = X_test[start:end]
        test_sq = np.sum(Xc ** 2, axis=1, keepdims=True)

        dists = test_sq + template_sq - 2.0 * Xc @ X_templates.T
        nn_idx = np.argmin(dists, axis=1)
        preds[start:end] = y_templates[nn_idx]

        print(f"Clustered NN chunk {end}/{X_test.shape[0]} done")

    return preds

#builds confusion matrixes 
def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

#calculates error rate
def error_rate(y_true, y_pred):
    return np.mean(y_true != y_pred)


# ----------------------------
# Task 2b: NN classifier using clustered templates
# ----------------------------

#timing the classification process using clustered templates
t1 = time.time()

#classiying
y_pred_clustered = nn_predict_chunked(testv, cluster_templates, cluster_template_labels, chunk=1000)

#calculates time
classification_time_clustered = time.time() - t1

#computes confusion matrix and error rate for the clustered template NN classifier
cm_clustered = confusion_matrix(testlab, y_pred_clustered, NUM_CLASSES)
err_clustered = error_rate(testlab, y_pred_clustered)

#prints it all
print("\n--- Task 2b: NN with clustered templates ---")
print("Confusion matrix (rows = true class, columns = predictions):")
print(cm_clustered)
print(f"\nError rate: {err_clustered * 100:.2f}% ({int(err_clustered * len(testlab))} of {len(testlab)} errors)")
print(f"Classification time with clustered templates: {classification_time_clustered:.2f} seconds")


#finds 7 nearest neighbors for each test sample
def knn_predict_chunked(X_test, X_templates, y_templates, K=7, chunk=1000):
    template_sq = np.sum(X_templates ** 2, axis=1)
    preds = np.empty(X_test.shape[0], dtype=np.int64)

    for start in range(0, X_test.shape[0], chunk):
        end = min(start + chunk, X_test.shape[0])
        Xc = X_test[start:end]
        test_sq = np.sum(Xc ** 2, axis=1, keepdims=True)

        # Squared Euclidean distances from current chunk to all templates
        dists = test_sq + template_sq - 2.0 * Xc @ X_templates.T

        # Indices of the K nearest templates for each test sample
        knn_idx = np.argpartition(dists, K-1, axis=1)[:, :K]

        # Majority vote among the K nearest labels
        for i in range(Xc.shape[0]):
            neighbor_labels = y_templates[knn_idx[i]]
            preds[start + i] = np.bincount(neighbor_labels, minlength=NUM_CLASSES).argmax()

        print(f"KNN chunk {end}/{X_test.shape[0]} done")

    return preds


# ----------------------------
# Task 2c: KNN classifier with K = 7
# ----------------------------
K = 7

#starts timing
t2 = time.time()

#starts classifying using KNN with clustered templates
y_pred_knn = knn_predict_chunked(testv, cluster_templates, cluster_template_labels, K=K, chunk=1000)

#calculating time 
classification_time_knn = time.time() - t2

#building confusion matrix and calculating error rate
cm_knn = confusion_matrix(testlab, y_pred_knn, NUM_CLASSES)
err_knn = error_rate(testlab, y_pred_knn)

print(f"\n--- Task 2c: KNN with clustered templates (K = {K}) ---")
print("Confusion matrix (rows = true class, columns = predictions):")
print(cm_knn)
print(f"\nError rate: {err_knn * 100:.2f}% ({int(err_knn * len(testlab))} of {len(testlab)} errors)")
print(f"Classification time with KNN: {classification_time_knn:.2f} seconds")