#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


#First look no dataframe
black_friday.head(5)


# In[15]:


#print nas dimensões do dataframe
print(black_friday.shape[0],"Linhas x", black_friday.shape[1], "Colunas")


# In[13]:


#Criação de lista com o nome das colunas
all_cols = list(black_friday.columns)
all_cols


# In[14]:


#Inspeção dos tipos das colunas
black_friday.dtypes


# In[16]:


#Averiguação da qtd de dados faltantes
black_friday.isna().sum()


# In[18]:


#Inspeção dos valores únicos em cada coluna
black_friday.nunique()


# In[19]:


#Categorias da coluna Age
black_friday['Age'].unique()


# In[49]:


#Auxílio para Questão 4
tipos = black_friday.dtypes #dtypes retorna variável do tipo Series
set_teste = set(tipos.values) #tranforma a Series em tipo Set(variável tipo conjunto com valores únicos apenas)
len(set_teste) #retorna o número de elementos na variável set_Teste


# In[32]:


bool(~black_friday['Product_Category_3'][(black_friday['Product_Category_2'].isnull())].notna().sum())#!= black_friday[(black_friday['Product_Category_2'].isnull()) & (black_friday['Product_Category_3'].notnull())].shape[0]


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[4]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[5]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return black_friday[(black_friday['Age'] == '26-35') & (black_friday['Gender'] == 'F')].shape[0]


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[6]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return black_friday['User_ID'].nunique()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[7]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return len(set(black_friday.dtypes.values))


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[8]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return float(black_friday.isnull().any(axis = 1).sum() / black_friday.shape[0])


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[9]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return max(black_friday.isnull().sum())


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[10]:


def q7():
    # Retorne aqui o resultado da questão 7.
    
    #cria uma lista com os valores não-nulos da coluna
    aux = list(black_friday['Product_Category_3'][black_friday['Product_Category_3'].notnull()]) 
    return max(set(aux), key = aux.count) #cria var. tipo set de acordo com ocorrência de cada elemento na lista


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[7]:


def q8():
    # Retorne aqui o resultado da questão 8.
    v = ((black_friday['Purchase'] - black_friday['Purchase'].min()) / 
         (black_friday['Purchase'].max() - black_friday['Purchase'].min()))
    return round(np.mean(v),3)


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[12]:


def q9():
    # Retorne aqui o resultado da questão 9.
    aux = list((black_friday['Purchase'] - black_friday['Purchase'].mean()) / (black_friday['Purchase'].std()))
    return sum([1 if i >= -1 and i <= 1 else 0 for i in aux])


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[13]:


def q10():
    # Retorne aqui o resultado da questão 10.
    
    #Numero de registros não-nulos de 'Product_Category_3' quando é nulo -> Se 0: afirmação verdadeira, do contrário falsa
    return bool(~black_friday['Product_Category_3'][(black_friday['Product_Category_2'].isnull())].notna().sum())

