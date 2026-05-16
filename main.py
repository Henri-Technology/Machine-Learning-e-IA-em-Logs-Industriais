# 1-IMPOTAÇÃO DAS BIBLIOTECAS

import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
import os
import time
import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv

# 2-IMPOTAÇÃO E ANÁLISE DOS DADOS BRUTOS

df = pd.read_csv('dataset.csv')

print('\n\n')
print(df.head())

print('\n\n')
print(df.describe())

print('\n\n')
print(df.isnull().sum())

print('\n\n')
print(df.duplicated().sum())

# Reflexão
print('\n')
print('''
---Reflexão---
      
Quais colunas apresentam valores faltantes?
R: As colunas que tem valores faltantes são: temperatura, vibracao e pressao

Esses dados faltantes precisam ser tratados antes do treinamento? Por quê?
R: Esses dados precisam ser tratados antes do treinamento, porque caso o modelo utilizar esses dados podemos obter erros de execução ou no pior dos casos, um código executado corretamente porém o resultado errado, por conta dos dados inconsistentes presentes no dataset utilizado para treino.''')



# 3-TRATAMENTO DE DADOS FALTANTES

df = df.fillna(df.mean())

print('\n\n')
print(df.isnull().sum())



# 4-INVESTIGAÇÃO DE OUTLIERS

# Criar boxplots para observar possíveis outliers
plt.figure(figsize=(10, 6))
df.boxplot()
plt.title("Boxplot das variáveis numéricas")
plt.ylabel("Valores")
# plt.show()

# Analise do aluno
print('\n')
print('''
---Analise do Aluno---
      
Quais colunas parecem ter outliers?
R: A coluna de tempo_operacao e temperatura parece ter outliers.

Esses outliers devem ser removidos ou apenas observados?
R: Depende, se a medida de tendência central a ser utilizada. Caso seja a média, sim os outliers podem ser removidos, porém, caso for mediana é recomendado mante-los.
      
Qual o impacto possível desses valores no modelo?
R: Caso utilizemos média para alguma análise ela pode ser diretamente impactada por Outliers, e quando o modelo for realizar a predição ele pode utilizar a medida de média comprometida pelos outliers, fazendo reduzir a precisão do modelo.''')



# 5-SEPARAÇÃO ENTRE VARIÁVEIS DE ENTRADA E ALVOS, E TREINO E TESTE

x = df.drop("falha", axis=1)
y = df['falha']

print('\n\n')
print("Formato de X:", x.shape)
print("Formato de y:", y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

print('\n\n')
print("X_train:", x_train.shape)
print("X_test:", x_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# Reflexão
print('\n')
print('''
---Reflexão---
      
Por que não devemos treinar e testar o modelo usando exatamente os mesmos dados??
R: Ele vai acertar tudo, e ele não vai conseguir saber se o modelo funciona com dados novos.''')



# 6-CRIAÇÃO E TREINAMENTO DO MODELO 

model = DecisionTreeClassifier(random_state=42)

model.fit(x_train, y_train)
print('\n\n')
print("Modelo treinado com sucesso.")



# 6-REALIZAR PREDIÇÕES

y_pred = model.predict(x_test)

print('\n\n')
print("Previsões:\n", y_pred)
print('\n')
print("Valores reais:\n", y_test.values)



# 7-MÉTRICAS DE DESEMPENHO

acc = accuracy_score(y_test, y_pred)

print('\n\n')
print(f"Acurácia do modelo: {acc:.2f}")

# Matriz de confusão 
print('\n\n')
matriz = confusion_matrix(y_test, y_pred)
print("Matriz de confusão:")
print(matriz)

# Relatório de classificação
print('\n\n')
print("Relatório de classificação:")
print(classification_report(y_test, y_pred))



# 8-VISUALIZAÇÃO DA VARIÁVEL ALVO

# Contagem da variável alvo
contagem_falha = df["falha"].value_counts()

# Gráfico de barras
plt.figure(figsize=(6, 4))
plt.bar(contagem_falha.index.astype(str), contagem_falha.values)
plt.title("Distribuição da variável alvo")
plt.xlabel("Falha (0 = normal, 1 = falha)")
plt.ylabel("Quantidade")
# plt.show()



# 9-CONCLUSÃO DO PROJETO

# Conclusão
print('\n')
print('''
---Conclusão---
      
O modelo teve bom desempenho?
R: Sim, teve uma precisão geral de 93%

Quais dificuldades foram encontradas no tratamento dos dados?
R: Foram encontradas, 2 principais dificuldades, a primeira sendo dados nulos, onde utilizei a tratativa de preenchimento utilizando os valores médios da coluna.
E a segunda foi a presença de alguns outliers no dataset, onde dependendo da análise realizada, pode alterar consideravelmente a precisão do valor final.

Os outliers podem ter influenciado o resultado?
R: Sim, pois a depender da análise realizada, o valor final pode ser consideravelmente influenciado pelos outliers no dataset utilizado.
  
Como esse tipo de solução pode ser útil em um ambiente industrial?
R: Pode ser útil para verificar a saúde e o desempenho de máquinas industriais, além de poder mostrar melhorias e previsões de falhas prováveis dessas máquinas.''')



# PT2 PROJETO

print('\n\n')
print("Parte 2 do Projeto")



# 2-FEATURE ENGINEERING

print('\n\n')
X_avancado = df.drop("falha", axis=1).copy()
y = df["falha"]

X_avancado["termica_pressao"] = X_avancado["temperatura"] * X_avancado["pressao"]

X_avancado["desgaste_acumulado"] = X_avancado["vibracao"] * X_avancado["tempo_operacao"]

print("Colunas atuais no dataset:", X_avancado.columns.tolist())
X_avancado.head()

print('''
---Reflexão do Aluno---
      
Pesquise e explique: Por que a criação de novas variáveis (Feature Engineering) baseadas no conhecimento de domínio (no caso, física industrial e mecânica) pode melhorar o poder preditivo de um modelo comparado ao uso de sensores puros?
R: Pois com essas novas variáveis relacionadas com os dados brutos, o modelo tem mais dados considerados "uteis" para treinar e testar, logo a sua precisão tende a aumentar.''')



# 3 - VALIDAÇÃO CRUZADA (CROSS-VALIDATION)

modelo_base_cv = DecisionTreeClassifier(random_state=42)

scores = cross_val_score(modelo_base_cv, X_avancado, y, cv=5, scoring='accuracy')

print("Acurácias por fold:", scores)
print(f"Acurácia média obtida: {scores.mean():.2%}")
print(f"Desvio padrão das acurácias: {scores.std():.4f}")



# 4 - OTIMIZAÇÃO DE HIPERPARÂMETROS (GRID SEARCH)

param_grid = {
    "max_depth": [3, 5, 7, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"]
}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_avancado, y)

print("Melhores Hiperparâmetros:", grid_search.best_params_)
print(f"Melhor acurácia de validação cruzada obtida: {grid_search.best_score_:.2%}")



# 5 - COMPARAÇÃO COM MODELOS ENSEMBLE: RANDOM FOREST

X_train_av, X_test_av, y_train_av, y_test_av = train_test_split(
    X_avancado, y, test_size=0.3, random_state=42
)

modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)

modelo_rf.fit(X_train_av, y_train_av)

y_pred_rf = modelo_rf.predict(X_test_av)

acc_rf = accuracy_score(y_test_av, y_pred_rf)
print(f"Acurácia do Random Forest: {acc_rf:.2%}")
print(classification_report(y_test_av, y_pred_rf))



# 6 - IMPORTÂNCIA DOS RECURSOS (FEATURE IMPORTANCE)

importancias = modelo_rf.feature_importances_

nomes_features = X_avancado.columns

df_importancia = pd.DataFrame({
    "Sensor": nomes_features,
    "Importância": importancias
}).sort_values(by="Importância", ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(df_importancia["Sensor"], df_importancia["Importância"])
plt.title("Importância dos Sensores e Recursos no Diagnóstico de Falhas")
plt.xlabel("Grau de Importância (0 a 1)")
plt.ylabel("Variáveis do Dataset")
plt.tight_layout()
plt.show()



# 7 - CONCLUSÃO DA SEGUNDA ETAPA

print('''
---Conclusão da Segunda Etapa---

1. As novas variáveis criadas (termica_pressao e desgaste_acumulado) agregaram valor preditivo?
R: Sim. A criação dessas variáveis combinadas permitiu que o modelo capturasse padrões de falha mais complexos,
representando fenômenos físicos reais como estresse térmico e desgaste progressivo. Isso contribui para
manter ou melhorar a acurácia em relação ao modelo básico da Etapa 1.

2. Qual a utilidade prática da Validação Cruzada em datasets pequenos (ex: 200 registros)?
R: Em datasets pequenos, uma única divisão treino/teste pode ser enviesada dependendo de quais amostras
caem em cada parte. A validação cruzada K-Fold resolve isso treinando e testando o modelo em 5 subconjuntos
diferentes, calculando a média e o desvio padrão das acurácias. Isso dá uma estimativa muito mais confiável
da real capacidade de generalização do modelo.

3. Como o Grid Search ajuda a garantir que o modelo não falhe com dados novos em tempo real?
R: O Grid Search testa sistematicamente diversas combinações de hiperparâmetros (como profundidade máxima
da árvore e número mínimo de amostras por folha) e seleciona os que resultam na melhor acurácia média
na validação cruzada. Com isso, evitamos o overfitting — situação onde o modelo "decora" os dados de treino
mas falha com dados novos — garantindo um modelo mais robusto para o ambiente de produção.

4. Qual sensor é o mais crítico para detecção precoce de falhas?
R: Com base no gráfico de importância dos recursos, a variável com maior peso é a mais crítica para o modelo.
Se fosse o engenheiro responsável, focaria o investimento em calibração constante nesse sensor, pois ele
é o principal gatilho para identificar anomalias antes que se tornem falhas graves na linha de produção.
''')


# EXTRA: EXEMPLO STREAMLIST

st.title("Monitoramento Industrial com IA")

st.dataframe(df)

st.line_chart(df["temperatura"])


# PT3 PROJETO

print('\n\n')
print("Parte 3 do Projeto")



# 1 - CONFIGURAÇÃO DA API KEY 

load_dotenv()
minha_chave = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=minha_chave)



# 2 - INICIALIZANDO O MODELO GEMINI

model = genai.GenerativeModel("gemini-2.5-flash-lite")



# 3 - INICIALIZANDO O MODELO GEMINI

response = model.generate_content("Explique rapidamente o que é manutenção preditiva industrial.")
print(response.text)



# 4 - SELECIONANDO UM REGISTRO

registro = df.iloc[10]

print(registro)



# 5 - CRIANDO UM PROMPT INTELIGENTE

prompt = f'''
Você é um especialista em manutenção industrial.

Analise os dados abaixo:

Temperatura: {registro['temperatura']}
Vibração: {registro['vibracao']}
Pressão: {registro['pressao']}
Tempo de operação: {registro['tempo_operacao']}

Explique:
1. Se existe risco de falha;
2. Qual sensor aparenta maior criticidade;
3. Sugestões de manutenção preventiva.
'''


# 6 - ENVIANDO DADOS AO GEMINI

response = model.generate_content(prompt)
time.sleep(4)
print(response.text)



# 7 - INTEGRAÇÃO COM MACHINE LEARNING

prompt_ml = f'''
O sistema de Machine Learning detectou:

Previsão de falha: {y_pred}

Explique:
- O que essa previsão significa;
- Quais ações devem ser tomadas;
- Possíveis riscos industriais.
'''

response = model.generate_content(prompt_ml)
time.sleep(4)
print(response.text)



# 8 - GERAÇÃO AUTOMÁTICA DE REALATÓRIOS

prompt_relatorio = f'''
Crie um relatório técnico resumido com base nos dados:

Temperatura: {registro['temperatura']}
Vibração: {registro['vibracao']}
Pressão: {registro['pressao']}
Tempo de operação: {registro['tempo_operacao']}

O relatório deve conter:
- Diagnóstico;
- Grau de risco;
- Recomendação técnica.
'''

response = model.generate_content(prompt_relatorio)
time.sleep(4)
print(response.text)



# 9 - ANALISANDO MÚLTIPLOS REGISTROS

for i in range(3):
    registro = df.iloc[i]

    prompt = f'''
    Analise os sensores industriais abaixo:

    Temperatura: {registro['temperatura']}
    Vibração: {registro['vibracao']}
    Pressão: {registro['pressao']}
    Tempo de operação: {registro['tempo_operacao']}

    Gere um diagnóstico técnico resumido.
    '''

    response = model.generate_content(prompt)
    time.sleep(4)
    print(f"===== REGISTRO {i} =====")
    print(response.text)
    print()



# 10 - SALVANDO RELATÓRIOS

with open("diagnostico.txt", "w", encoding="utf-8") as arquivo:

    for i in range(5):
        
        registro = df.iloc[i]

        prompt = f'''
        Analise os sensores:

        Temperatura: {registro['temperatura']}
        Vibração: {registro['vibracao']}
        Pressão: {registro['pressao']}
        Tempo de operação: {registro['tempo_operacao']}

        Gere um diagnóstico técnico.
        '''

        response = model.generate_content(prompt)
        time.sleep(4) # Espera 4 segundos entre cada registro

        arquivo.write(f"===== REGISTRO {i} =====\n")
        arquivo.write(response.text)
        arquivo.write("\n\n")

print("Relatório salvo com sucesso!")