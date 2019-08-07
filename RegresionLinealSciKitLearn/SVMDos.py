"""
Vectores de Soporte Regresión

@author: ligdigonzalez
"""

########## LIBRERÍAS A UTILIZAR ##########

#Se importan la librerias a utilizar
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


########## PREPARAR LA DATA ##########

#Importamos los datos de la misma librería de scikit-learn
boston = datasets.load_boston()
print(boston)
print()

########## ENTENDIMIENTO DE LA DATA ##########

#Verifico la información contenida en el dataset
print('Información en el dataset:')
print(boston.keys())
print()

#Verifico las características del dataset
print('Características del dataset:')
print(boston.DESCR)

#Verifico la cantidad de datos que hay en los dataset
print('Cantidad de datos:')
print(boston.data.shape)
print()

#Verifico la información de las columnas
print('Nombres columnas:')
print(boston.feature_names)

########## PREPARAR LA DATA VECTORES DE SOPORTE REGRESIÓN ##########

#Seleccionamos solamente la columna 6 del dataset
X_svr = boston.data[:, np.newaxis, 5]

#Defino los datos correspondientes a las etiquetas
y_svr = boston.target
print("AQUIIIIIIIII")
print(y_svr)

#Graficamos los datos correspondientes
plt.scatter(X_svr, y_svr)
plt.show()

########## IMPLEMENTACIÓN DE VECTORES DE SOPORTE REGRESIÓN ##########

from sklearn.model_selection import train_test_split

#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X_svr, y_svr, test_size=0.2)

from sklearn.svm import SVR

#Defino el algoritmo a utilizar
svr = SVR(kernel='linear', C=1.0, epsilon=0.2)

#Entreno el modelo
svr.fit(X_train, y_train)

#Realizo una predicción
Y_pred = svr.predict(X_test)

#Graficamos los datos junto con el modelo
plt.scatter(X_test, y_test)
plt.plot(X_test, Y_pred, color='red', linewidth=3)
plt.show()

print()
print('DATOS DEL MODELO VECTORES DE SOPORTE REGRESIÓN')
print()

print('Precisión del modelo:')
print(svr.score(X_train, y_train))