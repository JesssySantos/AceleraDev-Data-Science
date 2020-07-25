#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[248]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (KBinsDiscretizer, MinMaxScaler, StandardScaler)
from sklearn.feature_extraction.text import (CountVectorizer, TfidfVectorizer)
from sklearn.datasets import fetch_20newsgroups


# In[207]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

#from IPython.core.pylabtools import figsize


#figsize(12, 8)

sns.set()


# In[208]:


countries = pd.read_csv("countries.csv")


# In[209]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[210]:


# Sua análise começa aqui.
countries.dtypes


# In[211]:


columns_str = countries.select_dtypes(['object']).columns.to_list() #Cria lista com colunas do tipo string
comma_columns = columns_str[2:] #Retira da lista os dois primeiros elementos: colunas Country e Region
comma_columns.remove('Climate') #Remove variável categórica Climate
print(comma_columns)


# In[212]:


#Loop que substitui vírgulas por pontos e converte os valores da coluna para float
for col in comma_columns:
    countries[col] = countries[col].str.replace(',','.')
    countries[col] = countries[col].astype(float)


# In[213]:


#Retira espaços do começo e fim da string
countries['Country'] = countries['Country'].str.strip()
countries['Region'] = countries['Region'].str.strip()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[260]:


def q1():
    return sorted(countries['Region'].unique()) #Retorna lista ordenada alfabeticamente de regiões únicas


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[216]:


def q2():
    discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile") #Instancia o discretizer
    discretizer.fit(countries[['Pop_density']])
    
    #Cria array com o bin correspondente de cada registro
    score_bins = pd.DataFrame(discretizer.transform(countries[['Pop_density']]),columns=['Quartil'])
    
    return len(countries.loc[score_bins[score_bins['Quartil'] == 9].index,'Country'].unique())


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[217]:


def q3():
    #Baseando-se no funcionamento do One-Hot Enconding, sabe-se que para cada categoria cria-se uma nova coluna(Atributo):
    return len(countries['Region'].unique()) + len(countries['Climate'].unique())


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[218]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[270]:


print(comma_columns)


# In[219]:


def q4():
    data_missing = countries.copy() #Cria cópia do dataframe countries
    
    #Instancia o pipeline com as operações desejadas
    num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())])

    #Retorna índice númerico da coluna Arable no dataframe apenas com colunas float64
    col_arable = data_missing[comma_columns].columns.get_loc('Arable')

    #Aplica-se o pipeline na cópia do dataframe countries apenas nas colunas numéricas
    num_pipeline.fit(data_missing[comma_columns])
    new_data = pd.DataFrame(columns=countries.columns) #Criação de um dataframe com mesmas colunas do df countries

    #Insere-se os dados da lista test_country como linha no novo dataframe criado
    new_data = new_data.append(pd.Series(test_country, index=new_data.columns), ignore_index=True)

    return float(num_pipeline.transform(new_data[comma_columns])[0][col_arable].round(3))


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[221]:


def q5():
    q1 = countries['Net_migration'].quantile(0.25)
    q3 = countries['Net_migration'].quantile(0.75)
    iqr = q3 - q1 #Cálculo do iqr
    non_outlier_interval_iqr = [q1 - 1.5 * iqr, q3 + 1.5 * iqr] #Range aceitável para não ser outlier
    outliers_abaixo = len(countries['Net_migration'][(countries['Net_migration'] < non_outlier_interval_iqr[0])])
    outliers_acima = len(countries['Net_migration'][(countries['Net_migration'] > non_outlier_interval_iqr[1])])
    return (outliers_abaixo, outliers_acima, False)


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[232]:


#Obtenção dos dados
categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[245]:


def q6():
    count_vectorizer = CountVectorizer()
    newsgroups_counts = count_vectorizer.fit_transform(newsgroup.data)
    return int(newsgroups_counts[:,count_vectorizer.vocabulary_['phone']].sum())


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[253]:


def q7():
    tfidf_transformer = TfidfVectorizer()
    newsgroups_counts = tfidf_transformer.fit_transform(newsgroup.data)
    return float(newsgroups_counts[:,tfidf_transformer.vocabulary_['phone']].sum().round(3))


# In[254]:


type(TfidfVectorizer())


# In[ ]:




