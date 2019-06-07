# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:20:46 2017

@author: alef1

Iris  PCA
"""
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

def main():
    
    #buscando os valores direto do site
    df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',')
    #definição dso nomes das colunas
    df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
    df.dropna(how="all", inplace=True) # drops the empty line at file-end

    df.tail()
    #definindo os valores de X e Y
    X = df.ix[:,0:4].values
    y = df.ix[:,4].values
    #label de classificação

    feature_dict = {0: 'sepal length [cm]', 1: 'sepal width [cm]', 2: 'petal length [cm]', 3: 'petal width [cm]'}
    #primeira etapa apenas uma visualisação das colunas para com as classes mostrandos mistruras que os dados podem conter e em quais variáveis
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(8, 6))
        for cnt in range(4):
            plt.subplot(2, 2, cnt+1)
            for lab in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):
                plt.hist(X[y==lab, cnt],
                     label=lab,
                     bins=10,
                     alpha=0.3,)
            plt.xlabel(feature_dict[cnt])
        plt.legend(loc='upper right', fancybox=True, fontsize=8)

        plt.tight_layout()
        plt.show()
    
    # Padronização dos valores com media 0 e desvio padrão 1
    entrada = StandardScaler().fit_transform(X)
    
    #matriz de covariancia, poderia ser usada uma matriz de correlação, podem daria o mesmo valor 
    mat_covariancia = np.mean(entrada, axis=0)
    #calculo da covariancia 
    cov_mat = (entrada - mat_covariancia).T.dot((entrada - mat_covariancia)) / (entrada.shape[0]-1)
    cov_mat = np.cov(entrada.T)
    #auto valores e auto vetores
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    
    u,s,v = np.linalg.svd(entrada.T)
    for ev in eig_vecs:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    
    
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

    #aleatorizando os auto valores e auto vetores
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

   
    matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1)))
    
    Y = entrada.dot(matrix_w)
    print("matrixx:",len(Y[0]))
    #visualização dos dados a partir dos componentes principais encontrados
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(6, 4))
        for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                        ('blue', 'red', 'green')):
            plt.scatter(Y[y==lab, 0],
                    Y[y==lab, 1],
                    label=lab,
                    c=col)
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.legend(loc='Centro Inferior')
        plt.tight_layout()
        plt.show()
    
if __name__ == "__main__":
    main()