# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 03:17:21 2019

@author: user
"""
#%% Importing libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns

#%% Reading file 
data = pd.read_csv("case_zonas-valor.csv") 
# first lines
print(data.head())
# structure: 1176675x27
# numero_do_contribuinte	tipo_de_contribuinte_1	numero_do_condominio	
# codlog_do_imovel	nome_de_logradouro_do_imovel	numero_do_imovel	
# complemento_do_imovel	bairro_do_imovel	referencia_do_imovel	cep_do_imovel
# fracao_ideal	area_do_terreno	area_construida	area_ocupada	valor_do_m2_do_terreno	
# valor_do_m2_de_construcao	ano_da_construcao_corrigido	quantidade_de_pavimentos
# testada_para_calculo	tipo_de_uso_do_imovel	tipo_de_padrao_da_construcao
# tipo_de_terreno	fator_de_obsolescencia	ano_de_inicio_da_vida_do_contribuinte
# mes_de_inicio_da_vida_do_contribuinte	latitude	longitude
# 
#%%
# select a row
print(data.iloc [[0]])
# select a column
print(data['fator_de_obsolescencia'])
# Select a cell row num x column num
print(data.iloc[0,10])
# drop a column
minimal = data.drop(columns=['bairro_do_imovel', 'referencia_do_imovel']) # NULL

#%%
# Reducing data
data_reduced = data.drop(columns=['numero_do_contribuinte','numero_do_condominio', 
                                  'codlog_do_imovel','nome_de_logradouro_do_imovel',
                                  'numero_do_imovel', 'complemento_do_imovel',
                                  'bairro_do_imovel',
                                  'referencia_do_imovel', 'cep_do_imovel',
                                  'ano_de_inicio_da_vida_do_contribuinte',
                                  'mes_de_inicio_da_vida_do_contribuinte']) # NULL
data_reduced.to_csv('case_zonas-valor-reduzido.csv')
print(data_reduced['fator_de_obsolescencia'])
dimension = data_reduced.shape
header = list(data_reduced.columns.values)
#%% data processing pt. 1
cols = ['fracao_ideal', 'valor_do_m2_do_terreno', 
        'valor_do_m2_de_construcao', 'testada_para_calculo', 'fator_de_obsolescencia']
#data_reduced[cols].str.replace(',','.')
for i in cols:
    data_reduced[i] =  data_reduced[i].str.replace(',','.')

#print(data_reduced['fracao_ideal'])
#%% Processing pt. 2
col1 = 'tipo_de_contribuinte_1'
print(data_reduced[col1].unique())
dic1 = {'PESSOA FISICA (CPF)': 0, 'PESSOA JURIDICA (CNPJ)': 1}
data_reduced[col1] = data_reduced[
        col1].replace(dic1)
print(data_reduced[col1])

# Remove NaN from table based on tipo_de_contribuinte
data_reduced = data_reduced[pd.notnull(data_reduced[col1])]
print(data_reduced[col1].unique())

#%% Processing pt. 3
col2 = 'tipo_de_uso_do_imovel'
print(data_reduced[col2].unique())
dic2 = {'Apartamento em condomínio': 0,
 'Garagem (unidade autônoma) em edifício em condomínio de uso exclusivamente residencial':1,
 'Flat residencial em condomínio':2,
 'Flat de uso comercial (semelhante a hotel)':3,
 'Garagem (unidade autônoma) em edifício em condomínio de escritórios, consultórios ou misto':4,
 'Escritório/consultório em condomínio (unidade autônoma)':5,
 'Loja em edifício em condomínio (unidade autônoma)':6,
 'Garagem (unidade autônoma) de prédio de garagens':7,
 'Loja e residência (predominância comercial)': 8,
 'Residência':9,
 'Prédio de escritório ou consultório, não em condomínio, com ou sem lojas':10}
data_reduced[col2] = data_reduced[
        col2].replace(dic2)

print(data_reduced[col2].unique())

#%% Processing pt. 4
col3 = 'tipo_de_padrao_da_construcao'
print(data_reduced[col3].unique())
dic3 = {'Residencial vertical - padrão C':0, 'Residencial vertical - padrão D':1,
 'Residencial vertical - padrão E':2, 'Residencial vertical - padrão F':3,
 'Residencial vertical - padrão B':4, 'Comercial vertical - padrão C':5,
 'Comercial vertical - padrão D':6, 'Comercial vertical - padrão E':7,
 'Comercial vertical - padrão B':8, 'Residencial horizontal - padrão B':9,
 'Comercial vertical - padrão A':10, 'Residencial horizontal - padrão D':11,
 'Residencial vertical - padrão A':12, 'Comercial horizontal - padrão A':13,
 'Edifício de garagens - padrão A':14, 'Residencial horizontal - padrão C':15,
 'Comercial horizontal - padrão B':16, 'Residencial horizontal - padrão E':17,
 'Residencial horizontal - padrão F':18, 'Oficina/Posto de serviço/Armazém/D':19,
 'Comercial horizontal - padrão C':20
        }
data_reduced[col3] = data_reduced[
        col3].replace(dic3)

print(data_reduced[col3].unique())

#%% Processing pt. 5
col4 = 'tipo_de_terreno'
print(data_reduced[col4].unique())
dic4 = {'Normal':1, 'De duas ou mais ':2, 'De esquina':3, 'Terreno interno':4,
 'Lote de esquina ':5, 'Lote de fundos':6
        }
data_reduced[col4] = data_reduced[
        col4].replace(dic4)
print(data_reduced[col4].unique())
data_reduced.to_csv('case_zonas-valor-reduzido.csv')
#%% Visualizing data
x_column = 'longitude'
y_column = 'latitude'
x = data_reduced[x_column].astype(float)
y = data_reduced[y_column].astype(float)
# setup the plot
fig, ax = plt.subplots(1,1, figsize=(6,6))

scatter = ax.scatter(x, y, alpha=0.3 ,edgecolors='none')
plt.xlabel(x_column)
plt.ylabel(y_column)
plt.show()

#%% Reducing dimension
#1. Standarization
data_reduced_values = data_reduced.astype(float)

scaler = preprocessing.StandardScaler()
data_standarized = scaler.fit_transform(data_reduced_values)
data_reduced_standarized = pd.DataFrame(data_standarized, columns=header)
#%%
# 2. Correlation heatmap
corr = data_reduced_standarized.corr()
plt.figure(2,figsize=(16,9))
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=10),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
#%%
data_reduced_standarized = data_reduced_standarized.drop(columns=['ano_da_construcao_corrigido'])
corr = data_reduced_standarized.corr()
data_reduced_standarized.to_csv('case_zonas-valor-reduzido-std.csv')
        
# Observa-se uma forte correlação entre o ano do imóvele o faotr de obsolescencia
# Opta-se por remover o ano do terreno (diminuir 1 dimensão)

#%%
# 3. Eigenvalues and eigenvectors
data_reduced_standarized = pd.read_csv('case_zonas-valor-reduzido-std.csv')
corr = data_reduced_standarized.corr()
eig_vals, eig_vecs = np.linalg.eig(corr)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Tudo certo!')

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Autovalores em ordem decrescente:')
for i in eig_pairs:
    print(i[0])
    
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

x = [i for i in range(1,eig_vals.size+1)]
y = var_exp
threshold = [95 for i in range(1,eig_vals.size+1)] # arbitrario
plt.figure(figsize=(8,5))
plt.bar(x,y)
plt.plot(x,cum_var_exp, c='green')
plt.plot(x,threshold, c='red')
plt.xlabel('Principal Components')
plt.ylabel('Explained variance in percent')
plt.legend(['Cumulative','Threshold', 'Data'])
plt.show()

# Conclui-se que 12 features ja representam 95% das informações contidas
#%%
# 4. Apply the PCA for 12 features
data_reduced_standarized = pd.read_csv('case_zonas-valor-reduzido-std.csv')
n = 12
sklearn_pca = sklearnPCA(n_components=n)
data_PCA = sklearn_pca.fit_transform(data_reduced_standarized)
pca_header = ['PC_%s' %i for i in range(1,n+1)]
data_PCA = pd.DataFrame(data_PCA, columns=pca_header)
data_PCA.to_csv('case_zonas-valor-reduzido-PCA-'+str(n)+'.csv')
corr2 = data_PCA.corr()

plt.figure(figsize=(16,9))
ax = sns.heatmap(
    corr2, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=10),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.show()


#%%
# 5. Gaussian MIXTURE
#The optimal number of clusters (K) is the value that minimizes the
# Akaike information criterion (AIC) or the Bayesian information criterion (BIC).

n = 12
data_reduced = pd.read_csv('case_zonas-valor-reduzido.csv')
data_reduced = data_reduced.drop(columns=['Unnamed: 0'])
data_reduced_sampled = data_reduced.sample(frac = 0.01)
data_reduced_sampled.to_csv('data-reduced-sampled.csv')
header = list(data_reduced.columns.values)
data_reduced_values = data_reduced_sampled.astype(float)
scaler = preprocessing.StandardScaler()
data_standarized = scaler.fit_transform(data_reduced_values)
data_reduced_standarized = pd.DataFrame(data_standarized, columns=header)

sklearn_pca = sklearnPCA(n_components=n)
data_PCA = sklearn_pca.fit_transform(data_reduced_standarized)
pca_header = ['PC_%s' %i for i in range(1,n+1)]
data_PCA_sampled = pd.DataFrame(data_PCA, columns=pca_header)

X = data_PCA_sampled
n_components = np.arange(100, 100,10)
MC = 1
y_bic_means = np.empty([MC,n_components.size])
y_aic_means = np.empty([MC,n_components.size])

for i in np.arange(0,MC):
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X) for n in n_components]
    y_bic_means[i,:] = np.array([m.bic(X) for m in models])
    y_aic_means[i,:] = np.array([m.aic(X) for m in models])

plt.figure(figsize=(15,8))
plt.plot(n_components, np.mean(y_bic_means,0), label='BIC')
plt.plot(n_components, np.mean(y_aic_means,0), label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')

#plt.plot(n_components, [m.bic(X) for m in models], label='BIC')


#%% Define labels
# escolha de 150 bairros 
n_components = 150
#model = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0).fit(X)
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
gmm.fit(X)
labels = gmm.predict(X)
data_reduced_sampled['labels'] = labels
data_reduced_sampled.to_csv('novos-bairros-sampled-0_01.csv')
plt.figure(figsize=(15,8))
plt.scatter(data_reduced_sampled['longitude'], data_reduced_sampled['latitude'], c=labels, cmap='viridis')


