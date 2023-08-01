from sklearn.model_selection import GridSearchCV, cross_val_predict, train_test_split
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, roc_curve, precision_recall_curve, classification_report, roc_auc_score
from sklearn import svm

import pandas as pd
import numpy as np

from lecture import alphabet, choix

def gaussian_tune(X, y):
  parametres_rbf = [
    {
        'gamma':[0.5, 0.1, 1, 5, 10],
        'C':[0.01, 0.1, 0.5, 1, 5, 10],
        'kernel':['rbf']
    }
  ]
  model_rbf = svm.SVC(cache_size = 1000)
  model_1 = GridSearchCV(model_rbf, parametres_rbf, cv = 3, n_jobs = -1, verbose = 2, scoring = 'f1')
  model_1.fit(X, y)
  print("Best parameters : ")
  print(model_1.best_params_)
  return model_1.best_estimator_

def poly_tune(X_train, y_train):
  parametres_poly = [
      {
          'gamma':[1, 0.1, 0.05, 0.01],
          'C':[0.01, 0.1, 1, 10],
          'degree':[1, 2, 3],
          'coef0': [1, 2],
          'kernel':['poly']
      }
  ]
  model_poly = svm.SVC(cache_size = 1000)
  model_2 = GridSearchCV(model_poly, parametres_poly, cv = 3, n_jobs = -1, verbose = 2, scoring = 'f1')
  model_2.fit(X_train, y_train)
  print("Best parameters : ")
  print(model_2.best_params_)
  return model_2.best_estimator_

def linear_tune(X_train, y_train):
  parametres_linear = [
      {
          'gamma':[0.0001, 0.001, 0.01, 0.1, 1, 10],
          'C':[0.005, 0.01, 0.1, 1, 10],
          'kernel':['linear']
      }
  ]
  model_linear = svm.SVC(cache_size = 1000)
  model_3 = GridSearchCV(model_linear, parametres_linear, cv = 3, n_jobs = -1, verbose = 2,scoring = 'f1')
  model_3.fit(X_train, y_train)
  print("Best parameters : ")
  print(model_3.best_params_)
  return model_3.best_estimator_

"""
model_1 = gaussian_tune(X_train, y_train)
y_s_1 = cross_val_predict(model_1, X_train, y_train, cv = 3, method = "decision_function")
fpr, tpr, thresholds = roc_curve(y_train, y_s_1)
precisions, recalls, thresholds = precision_recall_curve(y_train, y_s_1)
"""

cols = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','B','Z','X','*', 'J', 'O', 'U']
Blosum62 = pd.DataFrame(0, index = cols , columns = cols)

Blosum62['A'] = [  4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1,  1, 0, -3, -2, 0, -2, -1, 0, -4, 0, 0, 0]
Blosum62['R'] = [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1, -4, 0, 0, 0]
Blosum62['N'] = [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2,  1, 0, -4, -2, -3, 3, 0, -1, -4, 0, 0, 0] 
Blosum62['D'] = [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1, -4, 0, 0, 0] 
Blosum62['C'] = [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4, 0, 0, 0] 
Blosum62['Q'] = [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1,  0, -3, -1,  0, -1, -2, -1, -2, 0, 3, -1, -4, 0, 0, 0] 
Blosum62['E'] = [-1, 0, 0, 2, -4, 2, 5, -2,  0, -3, -3,  1, -2, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4, 0, 0, 0] 
Blosum62['G'] = [ 0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -2, -1, -4, 0, 0, 0] 
Blosum62['H'] = [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1, -4, 0, 0, 0]
Blosum62['I'] = [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2, -3,  1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -1, -4, 0, 0, 0]
Blosum62['L'] = [-1, -2,-3, -4,-1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1, 1, -4, -3, -1, -4, 0, 0, 0] 
Blosum62['K'] = [-1,  2, 0, -1, -3, 1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2, 0, 1, -1, -4, 0, 0, 0] 
Blosum62['M'] = [-1, -1,-2, -3, -1, 0, -2, -3, -2,  1,  2, -1,  5, 0, -2, -1, -1, -1, -1, 1, -3, -1, -1, -4, 0, 0, 0] 
Blosum62['F'] = [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2, 1, 3, -1, -3, -3, -1, -4, 0, 0, 0] 
Blosum62['P'] = [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4, 0, 0, 0] 
Blosum62['S'] = [1, -1, 1, 0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4, 1, -3, -2, -2, 0, 0, 0, -4, 0, 0, 0] 
Blosum62['T'] = [0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1,  5, -2, -2, 0, -1, -1, 0, -4, 0, 0, 0] 
Blosum62['W'] = [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -3, -2, -4, 0, 0, 0] 
Blosum62['Y'] = [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -2, -1, -4, 0, 0, 0] 
Blosum62['V'] = [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2,  1, -1, -2, -2, 0, -3, -1, 4, -3, -2, -1,-4, 0, 0, 0] 
Blosum62['B'] = [-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4,  0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1, -4, 0, 0, 0] 
Blosum62['Z'] = [-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1, 0, -1, -3, -2, -2, 1,  4, -1, -4, 0, 0, 0] 
Blosum62['X'] = [0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1, -4, 0, 0, 0] 
Blosum62['*'] = [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1, 0, 0, 0]

def similarity(u, v):
    return sum([Blosum62.loc[u[k], v[k]] for k in range(u.shape[0])])

bijection = pd.DataFrame(np.arange(26), index = alphabet, columns = ['Nombres'])

def custom(data, p, q):
    temp = data.copy()
    m = temp.shape[0]
    for i in range(m):
        indice = temp.loc[i, 'Indice droit']
        temp.loc[i, 'Primary structure'] = temp.loc[i, 'Primary structure'][indice - p : indice+q]
    for i in range(-p, q):
        temp[str(i)] = 0
    for i in range(m):
        for k in range(-p, q):
            #temp.loc[i, str(k)] = temp.loc[i, 'Primary structure'][k+p]
            temp.loc[i, str(k)] = bijection.loc[temp.loc[i, 'Primary structure'][k+p], 'Nombres']
    temp  = temp.drop(columns = ['Primary structure', 'Longueur', 'Indice droit', 'Cleavage site', 'Description'], errors = 'ignore')
    return temp

def total_custom(data, a, b):
  temp1 = custom(data, a , b)
  temp2 = custom(choix(data, a, b), a, b)
  temp1['Label'] = 1
  temp2['Label'] = 0
  resultat = pd.concat([temp1, temp2]).astype(int)
  return resultat#.reset_index(drop = True) 

def to_letter_array(X):
  f = lambda x : alphabet[int(x)]
  vf = np.vectorize(f)
  return vf(X)

def kernel_z(u, v):
    a1 = to_letter_array(u) 
    b1 = to_letter_array(v)
    m = a1.shape[0]
    w = b1.shape[0]
    n = a1.shape[1]
    A = np.zeros((m, w))
    resultat = 0
    for i in range(m):
        for j in range(w):
            #print("i : "+str(i)+", j : "+str(j))
            u = a1[i,:] 
            v = b1[j,:] 
            A[i,j] = similarity(u,v)
        if(i%100 == 0):
          print("ligne : "+str(i))
    return A

def corrected_freq(x, y, i, S):
    resultat = 0
    if x != y:
        resultat = S.loc[x, i] + S.loc[y, i]
    else :
        resultat = S.loc[x, i] + np.log(1 + np.exp(S.loc[x, i]))
    return resultat

def kernel_freq(a1, b1, S):
    p = -int(S.columns[0])
    m = a1.shape[0]
    n = b1.shape[0]
    A = np.zeros((m, n))
    d = a1.shape[1]
    for i in range(m):
        for j in range(n):
            u = a1[i,:]
            v = b1[j,:]
            resultat = 0
            for k in range(d):
                caractere_1 = alphabet[int(u[k])] 
                caractere_2 = alphabet[int(v[k])]
                resultat = resultat + corrected_freq(caractere_1, caractere_2, k - p, S)
            A[i, j] = np.exp(resultat)
        if(i%100 == 0):
          print("ligne : "+str(i))
    return A
