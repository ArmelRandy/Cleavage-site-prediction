import pandas as pd
import numpy as np
import plotly.graph_objects as go

alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
parametre_log = 1

def cleaning(dataset):
    # repartition en trois colonnes
    m = int(dataset.shape[0]/3)
    description = []
    primary_structure = []
    cleavage_site = []
    for i in range(m):
        description.append(dataset.iloc[3*i, 0])
        primary_structure.append(dataset.iloc[3*i+1, 0])
        cleavage_site.append(dataset.iloc[3*i+2, 0])
    resultat = pd.DataFrame(list(zip(description, primary_structure, cleavage_site)), columns = 
                           ['Description', 'Primary structure', 'Cleavage site'])
    return resultat

def add_column(dataset):
    # ajout des colonnes taille et l'indice à la droite du cleavage site 
    data = dataset.copy()
    data['Longueur'] = [len(data['Primary structure'][i]) for i in range(data.shape[0])]
    data['Indice droit'] = [data['Cleavage site'][i].find('C') for i in range(data.shape[0])]
    data = data.drop(columns = ['Description', 'Cleavage site'])
    return data

def indices(data):
    min_ = data['Indice droit'].min()
    max_ = data['Longueur'] - data['Indice droit'].min()
    return min_, max_

def counting(data, p, q):
    # occurence normalisée de chaque caractère à chaque position
    # somme sur une ligne  = somme sur une colonne = 1
    # F
    colonnes = [i for i in range(-p, q)]
    data_test = pd.DataFrame(0, index = alphabet, columns = colonnes)
    for caractere in alphabet:
        for i in colonnes :
            somme = 0
            for k in range(data.shape[0]):
                mot = data['Primary structure'][k]
                indice = data['Indice droit'][k]
                if(mot[indice+i] == caractere):
                    somme = somme + 1
            data_test.loc[caractere, i] = somme
    return (data_test + parametre_log)/(data.shape[0] + parametre_log*len(alphabet))

def occurence_compte(data):
    # Frequence de chaque caractère dans le dataset
    data_occurence = pd.DataFrame(0, index = alphabet, columns = ['Compte'])
    for caractere in alphabet :
        somme = 0
        for k in range(data.shape[0]):
            mot = data['Primary structure'][k][1:] 
            somme = somme + mot.count(caractere)
        data_occurence.loc[caractere, 'Compte'] = somme
    data_occurence = (data_occurence + parametre_log)
    return data_occurence

def occurence_simple(data):
    # Frequence de chaque caractère dans le dataset
    # G
    data_occurence = pd.DataFrame(0, index = alphabet, columns = ['Compte'])
    for caractere in alphabet :
        somme = 0
        for k in range(data.shape[0]):
            mot = data['Primary structure'][k][1:] 
            somme = somme + mot.count(caractere)
        data_occurence.loc[caractere, 'Compte'] = somme
    data_occurence = (data_occurence + parametre_log)/(np.sum(data['Longueur']) - data.shape[0] + parametre_log*len(alphabet))
    return data_occurence

def finale(F, G):
  # S
  return np.log(F) - np.log(G).values

def preprocessing(data, p, q):
  # dataset apres add_column, donc ses colonnes sont ['Primary structure','Indice droit','Longueur'] et retourne S
  F = counting(data, p, q)
  G = occurence_simple(data)
  S = finale(F, G)
  return S

##################################################################################################################################################################
def find_indice(caractere):
    #renvoie un vecteur (26, 1)
    for i in range(len(alphabet)):
        if caractere == alphabet[i] :
            indice = i
    retour = np.zeros((len(alphabet), 1))
    retour[indice] = 1
    return retour

def find_representation(mot):
    resultat = find_indice(mot[0])
    for i in range(1, len(mot)):
        resultat = np.concatenate((resultat, find_indice(mot[i])), axis = 0)  
    return resultat.T

def one_hot_matrix(data, p, q):
    resultat = pd.DataFrame(0, index = np.arange(data.shape[0]), columns = np.arange((p+q)*len(alphabet)))
    for i in range(data.shape[0]):
        indice = data['Indice droit'][i]
        mot = data['Primary structure'][i]
        matrice = find_representation(mot[indice-p:indice+q])
        for j in range(resultat.shape[1]):
            resultat.iloc[i][j] = matrice[0][j]
    return resultat
    
def choix (data_, p, q):
    # pour l'entrainement
    n = data_.shape[0]
    data = data_.copy()
    data = data.drop(columns = ['Cleavage site'], errors = 'ignore')
    for i in range(n):
        pr = data.loc[i, 'Indice droit']
        l = data.loc[i, 'Longueur']
        v = np.arange(p, l-q+1)
        v = np.delete(v, pr-p)
        data.loc[i, 'Indice droit'] = np.random.choice(v, 1)[0]
    return data

def split(X):
  return  X.drop(columns = ['Label']), X['Label']


def plot_roc_curve(fpr, tpr, label = None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = fpr, y = tpr))
    fig.add_trace(go.Scatter(x = [0, 1], y = [0, 1]))
    fig.update_xaxes(title_text = 'False Positive Rate', range = [0, 1])
    fig.update_yaxes(title_text = 'True Positive Rate', range = [0, 1])

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = thresholds, y = precisions[:-1], name = "Precision"))
    fig.add_trace(go.Scatter(x = thresholds, y = recalls[:-1], name = "Recall"))
    fig.update_xaxes(title_text = "Threshold")
    fig.update_yaxes(range = [0, 1])

