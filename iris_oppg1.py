import numpy as np

# --- Load class data (each file has 50 samples x 4 features) ---
class_1 = np.loadtxt("Iris files/class_1", delimiter=",")  # Setosa
class_2 = np.loadtxt("Iris files/class_2", delimiter=",")  # Versicolor
class_3 = np.loadtxt("Iris files/class_3", delimiter=",")  # Virginica

N_TRAIN = 30
N_TEST = 20

# --- Split each class: first 30 train, last 20 test ---
train_1, test_1 = class_1[:N_TRAIN], class_1[N_TRAIN:]
train_2, test_2 = class_2[:N_TRAIN], class_2[N_TRAIN:]
train_3, test_3 = class_3[:N_TRAIN], class_3[N_TRAIN:]

# --- Combine into full train/test feature matrices ---
x_train = np.vstack([train_1, train_2, train_3])  # (90, 4)
x_test = np.vstack([test_1, test_2, test_3])       # (60, 4)


# --- Labels: 0 = Setosa, 1 = Versicolor, 2 = Virginica ---
y_train = np.array([0]*N_TRAIN + [1]*N_TRAIN + [2]*N_TRAIN)
y_test = np.array([0]*N_TEST + [1]*N_TEST + [2]*N_TEST)

# Name lookups
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
CLASS_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

NUM_CLASSES = 3


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def one_hot(labels, num_classes):
    """Convert integer labels to one-hot target vectors (Eq. 19 requires this)."""
    t = np.zeros((len(labels), num_classes))
    t[np.arange(len(labels)), labels] = 1.0
    return t


def add_bias(X):
    """Augment feature matrix with a column of ones (absorbs w_o into W)."""
    return np.hstack([X, np.ones((X.shape[0], 1))])


def train_linear_classifier(X, targets, alpha=0.01, max_iter=10000, tol=1e-6):
    """
    MSE-trained linear classifier with sigmoid (Compendium ch. 3.2).

    W has shape (C, D+1) where D+1 includes the bias column.
    Uses batch gradient descent (Eq. 22 & 23).
    """
    N, D = X.shape
    C = targets.shape[1]

    W = np.random.randn(C, D) * 0.01
    mse_history = []

    for iteration in range(max_iter):
        # Forward pass: z_k = W @ x_k  ->  g_k = sigmoid(z_k)
        Z = X @ W.T                    # (N, C)
        G = sigmoid(Z)                 # (N, C)

        # MSE (Eq. 19)
        error = G - targets            # (N, C)
        mse = 0.5 * np.sum(error ** 2) / N
        mse_history.append(mse)

        # Gradient (Eq. 22): sum_k [(g_k - t_k) ∘ g_k ∘ (1-g_k)] x_k^T
        grad_local = error * G * (1 - G)          # (N, C)  element-wise
        grad_W = grad_local.T @ X / N             # (C, D)

        # Update (Eq. 23)
        W = W - alpha * grad_W

        if iteration > 0 and abs(mse_history[-2] - mse) < tol:
            print(f"Converged at iteration {iteration}, MSE = {mse:.6f}")
            break

    return W, mse_history


def predict(X, W):
    """Classify samples: pick the class with highest sigmoid output."""
    G = sigmoid(X @ W.T)
    return np.argmax(G, axis=1)


def confusion_matrix(y_true, y_pred, num_classes):
    """Row = true class, column = predicted class."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    return cm


def error_rate(y_true, y_pred):
    return np.mean(y_true != y_pred)


# --- Prepare data for the linear classifier ---
X_train = add_bias(x_train)
X_test = add_bias(x_test)
T_train = one_hot(y_train, NUM_CLASSES)

# --- Train ---
alpha = 0.5
W, mse_hist = train_linear_classifier(X_train, T_train, alpha=alpha, max_iter=5000)

# --- Evaluate ---
pred_train = predict(X_train, W)
pred_test = predict(X_test, W)

cm_train = confusion_matrix(y_train, pred_train, NUM_CLASSES)
cm_test = confusion_matrix(y_test, pred_test, NUM_CLASSES)

print("\n--- Training set ---")
print("Confusion matrix:")
print(cm_train)
print(f"Error rate: {error_rate(y_train, pred_train) * 100:.1f}%")

print("\n--- Test set ---")
print("Confusion matrix:")
print(cm_test)
print(f"Error rate: {error_rate(y_test, pred_test) * 100:.1f}%")