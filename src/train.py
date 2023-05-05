from network import Network
from preprocess import load_data, preprocess_data

def train(df, method='Default', compare=False):
    if method == 'Adam_KFold':
        kf_gen = preprocess_data(df, k_fold=True)
    else:
        X_train, X_val, y_train, y_val = preprocess_data(df, k_fold=False)

    network = Network([26, 24, 24, 2], compare=compare)
    if method == 'Default':
        network.train_default(X_train, y_train, X_val, y_val, epochs=250, lr=0.01)
    elif method == 'SGD':
        network.SGD(X_train, y_train, X_val, y_val, epochs=250, lr=0.03, batch_size=32)
    elif method == 'NAG':
        network.NAG(X_train, y_train, X_val, y_val, epochs=500, lr=0.01, batch_size=32, mu=0.4, early_stopping=True)
    elif method == 'RMSProp':
        network.RMSProp(X_train, y_train, X_val, y_val, epochs=500, lr=0.01, batch_size=32, beta=0.85, epsilon=1e-8, early_stopping=True)
    elif method == 'Adam':
        network.Adam(X_train, y_train, X_val, y_val, epochs=1000, lr=0.01, batch_size=32, early_stopping=True)
    elif method == 'Adam_KFold':
        network.Adam_KFold(kf_gen, epochs=1000, lr=0.01, batch_size=32, early_stopping=True)
    else:
        raise ValueError(f"Method {method} is not supported.")
    network.save('../assets/')

if __name__ == '__main__':
    train_df = load_data('../assets/data.csv')
    train(train_df, 'Adam_KFold')

