from network import Network
from preprocess import load_data, preprocess_data

if __name__ == '__main__':
    train_df = load_data('../assets/data.csv')

    X_train, X_val, y_train, y_val = preprocess_data(train_df, k_fold=False)
    # kf_gen = preprocess_data(train_df, k_fold=True)

    network = Network([26, 24, 24, 2], patience=15, min_delta=0.00005)
    network.SGD(X_train, y_train, X_val, y_val, epochs=10, lr=0.03, batch_size=16)
    # # network.NAG(X_train, y_train, X_val, y_val, epochs=1000, lr=0.03, batch_size=32, mu=0.4, early_stopping=True)
    # # network.RMSProp(X_train, y_train, X_val, y_val, epochs=1000, lr=0.05, batch_size=32, beta=0.85, epsilon=1e-8, early_stopping=True)
    # # network.Adam(X_train, y_train, X_val, y_val, epochs=1000, lr=0.03, batch_size=32, early_stopping=True)
    # network.Adam_KFold(kf_gen, epochs=1000, lr=0.01, batch_size=32, early_stopping=True)

    network.save('../assets/weights')
