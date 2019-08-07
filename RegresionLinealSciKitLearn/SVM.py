import pandas as pd
from sklearn.model_selection  import  train_test_split

import matplotlib.pyplot as plt

datos = pd.read_csv('wineq.csv')
datos= datos.astype(float).fillna(0.0)



X=datos.drop('quality',axis=1)


y=datos.quality


print("_________________")
print(X)
print("::::::::::::::::::::")
print(y)
print("_________________")
plt.scatter(datos.pH, y)




print(datos['quality'].value_counts())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_array = sc.fit_transform(X_train.values)
X_train = pd.DataFrame(X_train_array, index=X_train.index, columns=X_train.columns)
X_test_array = sc.transform(X_test.values)
X_test = pd.DataFrame(X_test_array, index=X_test.index, columns=X_test.columns)


from sklearn.svm import SVC
clf=SVC(kernel='rbf').fit(X_train,y_train) #linear,poly,rbf
print(clf.score(X_test,y_test)) #Entre mas cercano a uno mejor

#Realizo una predicción
Y_pred = clf.predict(X_test)

#print(Y_pred)


#Graficamos los datos junto con el modelo
#plt.scatter(X_test, y_test)
plt.scatter(datos.pH, y)
#plt.plot(X_test, Y_pred, color='red', linewidth=3)
plt.plot(X_test, Y_pred, color='red', linewidth=3)
plt.show()


