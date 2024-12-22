import pandas as pd
import numpy as np
import math
from collections import Counter
import csv
import os

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#On normalise les valeurs afin d'avoir moins de différences d'importances des distances euclediennes entre chaque points 

X_train=(train[['C1','C2','C3','C4','C5','C6','C7']]-train[['C1','C2','C3','C4','C5','C6','C7']].min())/(train[['C1','C2','C3','C4','C5','C6','C7']].max()-train[['C1','C2','C3','C4','C5','C6','C7']].min())
y_train=train['Label'].values

X_test=(test[['C1','C2','C3','C4','C5','C6','C7']]-train[['C1','C2','C3','C4','C5','C6','C7']].min())/(train[['C1','C2','C3','C4','C5','C6','C7']].max()-train[['C1','C2','C3','C4','C5','C6','C7']].min())
id_test=test['Id'].values

# On definit les différentes fonctions qui vont nous servir pour notre KNN
def distance_euclidienne_np(point1,point2):
    return np.linalg.norm(point1-point2)

#Ici on cherche tous les k voisins d'un point en calculant les distances eucledienne de chacun des points par rapport aux autres
def obtenir_les_voisins(X_train,x_test,k):
    distances_x_test=np.linalg.norm(X_train-x_test,axis=1)
    indices_voisins=np.argsort(distances_x_test)[:k]
    return y_train[indices_voisins]

#On prends la classe/ le label qui revient le plus dans les k voisins
def predire(X_train, x_test, k):
    voisins=obtenir_les_voisins(X_train,x_test,k)
    frequence_classe=Counter(voisins)
    label=frequence_classe.most_common(1)[0][0]
    return label


def predire_labels(X_train,X_test,k):
    return [predire(X_train,x_test,k) for x_test in X_test.values]


k = 2
predictions=predire_labels(X_train, X_test, k)

resultats=[(a,b) for a,b in zip(id_test, predictions)]

print(resultats)

fichier_resultats_csv='resultats_dernier_test.csv'

with open(fichier_resultats_csv,'w', newline='') as fichier:
    writer=csv.writer(fichier)
    writer.writerow(["Id","Label"])
    writer.writerows(resultats)

print(len(resultats))
