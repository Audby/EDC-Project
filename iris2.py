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

X_train_reduced = np.delete(X_train, worst_feature_index, axis=1)
X_test_reduced = np.delete(X_test, worst_feature_index, axis=1)

print("Removed feature:", feature_names[worst_feature_index])
print("New X_train shape:", X_train_reduced.shape)
print("New X_test shape:", X_test_reduced.shape)


