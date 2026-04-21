import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix

#loading the iris dataset
iris = load_iris()
X = iris.data
y = iris.target 
feature_names = iris.feature_names
class_names = iris.target_names

#the great split 
train_idx = []
test_idx = []

for i in range(3):
    class_idx = np.where(y==i)[0]
    train_idx.extend(class_idx[:30])
    test_idx.extend(class_idx[30:])

train_idx = np.array(train_idx)
test_idx = np.array(test_idx)

X_train = X[train_idx]
y_train = y[train_idx]
X_test = X[test_idx]
y_test = y[test_idx]

#histograms

for f in range(4):
    plt.figure(figsize=(8,5))
    for c in range(3):
        plt.hist(
            X_train[y_train==c, f],
            bins = 10,
            alpha = 0.5,
            label = class_names[c],
            density=True

        )
    plt.title("Histogram of feature: " + feature_names[f])
    plt.xlabel(feature_names[f])
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
   # plt.show()

    #the part above is for visualizing the distribution of each feature for the three classes in the training set.
    #this helps ut identify which features are more discriminative for classifying the iris species.
    #A feature is good it if shows distinct distributions for different classes, making it easier for a classifier to separate them.
    #A feature is bad if the distributions for different classes overlap significantly, making it harder for a classifier to distinguish between them.
    #The four histograms plotted shows exactly this for the four different features 


#calculating the histogram overlap score for each feature to quantitatively assess their discriminative power.
def histogram_overlap(feature_values, class_labels, bins=20):
    print("feature_values shape:", np.shape(feature_values))
    print("class_labels shape:", np.shape(class_labels))

    min_val = np.min(feature_values)
    max_val = np.max(feature_values)

    histogram_list = []

    for c in range(3):
        class_values = feature_values[class_labels == c]
        hist, edges = np.histogram(class_values, bins=bins, range=(min_val, max_val), density=True)
        histogram_list.append(hist)

    bin_width = edges[1] - edges[0]

    total_overlap = 0
    for i in range(3):
        for j in range(i+1, 3):
            total_overlap += np.sum(np.minimum(histogram_list[i], histogram_list[j])) * bin_width

    return total_overlap


overlap_scores = []

for f in range(X_train.shape[1]):
    score = histogram_overlap(X_train[:, f], y_train, bins=20)
    overlap_scores.append(score)
    print(f"Feature: {feature_names[f]}, Overlap Score: {score:.4f}")

worst_feature_index = np.argmax(overlap_scores)

print(f"Worst feature: {feature_names[worst_feature_index]}, Overlap Score: {overlap_scores[worst_feature_index]:.4f}")

#checking current shape
print("X_train shape before removing feature:", X_train.shape)
print("X_test shape before removing feature:", X_test.shape)

#removing the worst feature

# X_train_reduced = np.delete(X_train, worst_feature_index, axis=1)
# X_test_reduced = np.delete(X_test, worst_feature_index, axis=1)

# print("Removed feature:", feature_names[worst_feature_index])
# print("New X_train shape:", X_train_reduced.shape)
# print("New X_test shape:", X_test_reduced.shape)

#removing second worst feature

# X_train_2feat = X_train[:, [2, 3]]
# X_test_2feat = X_test[:, [2, 3]]

# print("Using features:", feature_names[2], "and", feature_names[3])
# print("X_train_2feat shape:", X_train_2feat.shape)
# print("X_test_2feat shape:", X_test_2feat.shape)

#removing the third worst feature

X_train_1feat = X_train[:, [3]]
X_test_1feat = X_test[:, [3]]

print("Using features:", feature_names[2], "and", feature_names[3])
print("X_train_1feat shape:", X_train_1feat.shape)
print("X_test_1feat shape:", X_test_1feat.shape)


#adding sigmoid function + more for the linear classifier

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def one_hot(labels, num_classes):
    t = np.zeros((len(labels), num_classes))
    t[np.arange(len(labels)), labels] = 1.0
    return t


def add_bias(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])


def train_linear_classifier(X, targets, alpha=0.01, max_iter=10000, tol=1e-6):
    """
    MSE-trained linear classifier with sigmoid
    """
    N, D = X.shape
    C = targets.shape[1]

    W = np.random.randn(C, D) * 0.01
    mse_history = []

    for iteration in range(max_iter):
        Z = X @ W.T
        G = sigmoid(Z)

        error = G - targets
        mse = 0.5 * np.sum(error ** 2) / N
        mse_history.append(mse)

        grad_local = error * G * (1 - G)
        grad_W = grad_local.T @ X / N

        W = W - alpha * grad_W

        if iteration > 0 and abs(mse_history[-2] - mse) < tol:
            print(f"Converged at iteration {iteration}, MSE = {mse:.6f}")
            break

    return W, mse_history


def predict(X, W):
    G = sigmoid(X @ W.T)
    return np.argmax(G, axis=1)


def error_rate(y_true, y_pred):
    return np.mean(y_true != y_pred)

#running classifier on the reduced feature set and evaluating performance

NUM_CLASSES = 3

X_train_cls = add_bias(X_train_1feat)
X_test_cls = add_bias(X_test_1feat)
T_train = one_hot(y_train, NUM_CLASSES)

alpha = 0.5
W, mse_hist = train_linear_classifier(X_train_cls, T_train, alpha=alpha, max_iter=5000)

pred_train = predict(X_train_cls, W)
pred_test = predict(X_test_cls, W)

cm_train = confusion_matrix(y_train, pred_train)
cm_test = confusion_matrix(y_test, pred_test)

train_err = error_rate(y_train, pred_train)
test_err = error_rate(y_test, pred_test)

# print("\n--- Reduced 1-feature classifier ---")
# print("Removed feature:", feature_names[worst_feature_index], "and", feature_names[worst_feature_index-1], "and", feature_names[worst_feature_index-3])

print("\n--- Reduced 1-feature classifier ---")
print("Used feature:", feature_names[3])

print("\nTraining confusion matrix:")
print(cm_train)
print(f"Training error rate: {train_err * 100:.2f}%")

print("\nTest confusion matrix:")
print(cm_test)
print(f"Test error rate: {test_err * 100:.2f}%")