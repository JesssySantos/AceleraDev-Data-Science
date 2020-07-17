#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[39]:


from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

from loguru import logger


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

#from IPython.core.pylabtools import figsize


#figsize(12, 8)

sns.set()


# In[44]:


fifa = pd.read_csv("fifa.csv")


# In[45]:


columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# ## Inicia sua análise a partir daqui

# In[46]:


# Sua análise começa aqui.
fifa.head()


# In[47]:


#Investigação de Missing Values
missing = fifa.isna().sum().to_frame(name='TOTAL')
missing['%'] = (missing['TOTAL']/fifa.shape[0])*100
missing.sort_values(by='%',ascending=False).head()


# In[48]:


fifa.dropna(inplace=True) #Apaga registros com missing values


# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# In[6]:


def q1():
    projected = PCA(n_components=1).fit(fifa) #Instaciando o objeto PCA com um componente e o aplicando
    return float(projected.explained_variance_ratio_.round(3)) #Retornando a variância esperada arredondada


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[19]:


def q2():
    pca_095 = PCA(0.95) #Instaciando o objeto PCA com 95% como fração de variância total
    X_reduced = pca_095.fit_transform(fifa) #Aplicando a redução
    return X_reduced.shape[1] #retornando o número de componentes principais(colunas)


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[20]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# In[36]:


def q3():
    projected = PCA().fit(fifa) #Instanciando e aplicando o objeto PCA mantendo todas componentes
    fifa_dot_x = projected.components_.dot(x) #Produto vetorial com vetor x
    coordenadas = fifa_dot_x[:2].round(3) #Retorna array com as duas primeiras componentes principais arredondas
    return tuple([float(num) for num in coordenadas]) #Converte as coordenadas para float e as retorna em tupla


# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# In[10]:


def q4():
    X = fifa.drop(columns='Overall') #Setando conjunto de features
    y = fifa['Overall'] #Setando variável target
    estimator = LinearRegression() #Atribuindo o tipo de estimador especificado
    selector = RFE(estimator, n_features_to_select=5).fit(X, y) #Criação e aplicação do modelo
    selected_features = selector.get_support(indices=True) #Retorna array de indices das features escolhidas
    return list(X.columns[selected_features]) #Retorna o nome das features escolhidas

