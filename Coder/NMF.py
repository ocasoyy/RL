from sklearn.decomposition import NMF
from sklearn.datasets import load_iris


iris = load_iris()
data = iris.data

nmf = NMF(n_components=2)
nmf.fit(data)
iris_nmf = nmf.transform(data)




