import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importando o modelo de aprendizado de máquina
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Importando a base de dados
# A base de dados contém informações sobre pinguins, incluindo características como espécie, tamanho e afins.

df_pinguins = pd.read_csv('penguins.csv')

# Verificando se a base de dados foi carregada corretamente
print("Base de dados carregada com sucesso!")

# Visualizando as primeiras linhas da base de dados
print(df_pinguins.head())

def preparar_dados(df):
    # Remover linhas com dados nulos
    df = df.dropna().copy()

    # Renomear colunas para facilitar o acesso
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    # Convertendo categorias em variáveis numéricas
    df['ilha'] = df['ilha'].astype('category').cat.codes
    df['sexo'] = df['sexo'].astype('category').cat.codes

    # Separando as features e o target
    X = df.drop(columns=['espece'])
    y = df['espece']
    
    return X, y

# Preparando os dados
X, y = preparar_dados(df_pinguins)

# Dividindo os dados em conjunto de treino e teste
# X são as features e y é o target
# 20% dos dados serão para teste (conforme foi definido no enunciado)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def treinar_modelo(X_train, y_train):
    # Criando o modelo de Random Forest
    # O n_estimators padrão é 100, mas pode ser ajustado conforme necessário
    # O random_state é definido para garantir reprodutibilidade
    modelo = RandomForestClassifier(random_state=42)
    
    # Treinando o modelo com os dados de treino
    modelo.fit(X_train, y_train)
    
    return modelo

def avaliar_modelo(modelo, X_test, y_test):
    # Avaliando o modelo com os dados de teste
    acuracia = modelo.score(X_test, y_test)
    print(f"Acurácia do modelo: {acuracia:.2f}%")
    
    return acuracia

def prever_especies(modelo, X):
    # Fazendo previsões com o modelo treinado
    previsoes = modelo.predict(X)
    
    return previsoes

def mostrar_resultados(previsoes, y_test):
    # Comparando as previsões com os valores reais
    resultados = pd.DataFrame({'Real': y_test, 'Previsto': previsoes})
    print(resultados.head())
    # Contando acertos e erros
    acertos = (resultados['Real'] == resultados['Previsto']).sum()
    erros = len(resultados) - acertos
    print(f"Acertos: {acertos}, Erros: {erros}")
    # Visualizando os resultados
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.barplot(x=resultados['Real'].value_counts().index, y=resultados['Real'].value_counts().values)
    plt.title('Valores Reais')
    plt.subplot(1, 2, 2)
    sns.barplot(x=resultados['Previsto'].value_counts().index, y=resultados['Previsto'].value_counts().values)
    plt.title('Valores Previsto')
    plt.show()
    # Salvando os resultados em um arquivo CSV
    resultados.to_csv('resultados_previsoes.csv', index=False, encoding='utf-8')
    print("Resultados salvos em 'resultados_previsoes.csv'.")

# Função principal para executar todo o processo
def executar_pipeline(df):
    X, y = preparar_dados(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = treinar_modelo(X_train, y_train)
    avaliar_modelo(modelo, X_test, y_test)
    previsoes = prever_especies(modelo, X_test)
    mostrar_resultados(previsoes, y_test)

executar_pipeline(df_pinguins)