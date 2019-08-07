import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


datos=pd.read_csv('movies_dos.csv')
print("AQUI: ")
print(datos)

df=pd.DataFrame(datos)
x=df[df.keys()[0]].values

y=df[df.keys()[1]].values

print("valor maximo de likes: ",df[df.keys()[0]].max())
print("valor minimo de likes: ",df[df.keys()[0]].min())
print("valor promedio de likes: ",df[df.keys()[0]].mean())

#Para la agrupacion primero se necesita tenerlos en un arreglo o matriz
#####info=df[[df.keys()[0],df.keys()[1]]].as_matrix()
#se muestra en una matriz
####print(info)
print()
print("forma de tratarlos en un array")
X=np.array(list(zip(x,y)))
print(X)

#agrupamiento

#cuantos grupos clusters
kmeans=KMeans(n_clusters=2)
#para ajustar los datos
kmeans=kmeans.fit(X)
#Asiganr etiquetas a los datos (automaticamente)  predicts esta basado en los centroides se estima la distancia menor entre el punto vecino
# esos a menor distancia los que se van agrupar
labels=kmeans.predict(X)
centroids= kmeans.cluster_centers_

colors=("m.","r.","c.","y.","g.")

#Rango en el linea 23 que es el array
for i in range(len(X)):
    print("Coordenada: ",X[i]," Label: ",labels[i])
    #primer columna de las x
    plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=10)
    #
#graficar los centrois el centro de

plt.scatter(centroids[:,0],centroids[:,1],marker='X',s=150,linewidths=5,zorder=10)
plt.show()