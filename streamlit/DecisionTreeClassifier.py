from sklearn.datasets import make_classification,make_blobs
import matplotlib.pyplot as plt

def load_initial_graph(dataset,ax):
    if dataset == "Binary":
        X, y = make_blobs(n_features=2, centers=2,random_state=6)
        ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
        return X,y
    elif dataset == "Multiclass":
        X,y = make_blobs(n_features=2, centers=3,random_state=2)
        ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
        return X,y
 
fig, ax = plt.subplots()
X,y = load_initial_graph("Binary",ax)
print(y)

a = []