# Simples implementação arvore de decisão - ID3
# é um algoritmo inventado por Ross Quinlan, usado para gerar uma árvore de decisão de um conjunto de dados.
# ID3 é o precursor do algoritmo C4.5, e é tipicamente usado nos domínios de aprendizagem mecânica e processamento de linguagem natural.

# bibliotecas

import math
import numpy as np
import random

# previsão de tempo
# para x1, temos 1= tempo nublado e 2= tempo com sol
# para x2, temos 0= ventos fortes e 1= ventos normais( dia estável)
x1 = [2, 1, 2, 1]
x2 = [1, 0, 1, 0]

# NumPy é um pacote de Python que suporta operações com vetores e matrizes e é essencial para a computação científica com Python.
y = np.array([0, 1, 1, 0])

# a ideia é que a partir de algumas regras se possa treinar e gerar uma classificação
# exemplo de um metodo que pode gerar um dicionário de dados de acordo com os índices
def divide_conjuntos(alfa):
    return {charlie: (alfa==charlie).nonzero()[0] for charlie in np.unique(alfa)}

# entropia, é uma medida de "desordem" em um sistema,  a entropia é calculada para cada atributo.
# O atributo com a menor entropia é usado para dividir o conjunto.
# Quanto maior a entropia, maior o potencial para melhorar a classificação aqui.
# métrica id3 disponível em: https://en.wikipedia.org/wiki/ID3_algorithm
# cálculo da pureza:

def entropia(eco):
    rs = 0
    values, countis = np.unique(eco, return_counts=True)
    frequencia = countis.astype('float')/len(eco)
    for papa in frequencia:
        if papa != 0.0:
            rs -= papa * np.log2(papa)
    return rs

#

def ganho(y, x):
# em arvore de decisao, a melhoria de pureza pode ser referida como ganho de informação
    rs = entropia(y)

    # Divisão de conjuntos, conforme atributos
    values, countis = np.unique(x, ret_countis=True)
    frequencia = countis.astype('float')/len(x)

    # cálculo da média ponderada
    for papa, vic in zip(frequencia, values):
        rs -= papa * entropia(y[x == vic])

    return rs
