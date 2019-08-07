import sklearn
import mglearn
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


mglearn.plots.plot_pca_illustration() # Ejes de los componentes pRo-datosd, 4 suma el promedio que se resto, -ruido

cancer=load_breast_cancer()
print("Nombres: ",cancer.feature_names)
print("Cuantas: ",cancer.feature_names.shape)

pca=PCA(n_components=2)  # dimensiones componenetes lineas mayor varianza c1, ortogonal angulo recto al c1 el c2
print(pca.fit(cancer.data)) # entrenar

transformada=pca.transform(cancer.data) # 30 dimensiones a 2
print("#################")
print("original: ",cancer.data.shape)
print("original: ",cancer.data)
print("_-_-_-_-_-_-_-_-_-_-_-")
print("Transformado 2D: ",transformada.shape)
print("Transformado 2D: ",transformada)
print("#################")

mglearn.discrete_scatter(transformada[:,0],transformada[:,1], cancer.target)
plt.legend(cancer.target_names,loc='best')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()



from sklearn.preprocessing import MinMaxScaler
escala=MinMaxScaler()
escala.fit(cancer.data)
escalada=escala.transform(cancer.data)
pca.fit(escalada)
transformada=pca.transform(escalada)
mglearn.discrete_scatter(transformada[:,0],transformada[:,1], cancer.target)
plt.legend(cancer.target_names,loc='best')
plt.gca()
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

print("Escala: ")
print(escalada)
print("_____________________")
print("Datos del cancer")
print(cancer.data)

#patrones
