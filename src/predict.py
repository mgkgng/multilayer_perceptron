import os, sys
import numpy as np
from network import Network
from preprocess import load_data, preprocess_data
from metrics import accuracy, precision, recall, f1_score, specificity

NUM_LAYERS = 4

def log_loss(y_true, y_pred, eps=1e-8):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def load_params(path):
    biases = []
    weights = []
    for i in range(NUM_LAYERS - 1):
        bfile = path + f"biases{i}.npy"
        wfile = path + f"weights{i}.npy"
        if not os.path.isfile(bfile) or not os.path.isfile(wfile):
            raise ValueError(f"File {bfile} or {wfile} does not exist. Finish the training first.")
        biases.append(np.load(bfile))
        weights.append(np.load(wfile))
    return biases, weights

def predict(X, y, y_prime=None, compare=False, epsilon=1e-8):
    biases, weights = load_params('../assets/')
    network = Network([26, 24, 24, 2], biases=biases, weights=weights)

    y_pred = network.predict(X)
    loss = log_loss(y, network.result(X).T, eps=epsilon)
    accuracy = accuracy(y, y_pred)
    precision = precision(y, y_pred)
    recall = recall(y, y_pred)
    f1 = f1_score(y, y_pred)
    spe = specificity(y, y_pred)

    if compare == False:
        print(f"Loss: {loss:.6f} | Accuracy: {accuracy * 100:.2f}% | Precision: {precision  * 100:.2f}% | Recall: {recall  * 100:.2f}% | F1 Score: {f1:.6f} | Specificity: {spe * 100:.2f}%")
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please specify the path to the data file as the first argument.")
    train_df = load_data(sys.argv[1])
    X, y = preprocess_data(train_df, predict=True)
    yy = np.argmax(y, axis=1)
    predict(X, yy, y_prime=y)
