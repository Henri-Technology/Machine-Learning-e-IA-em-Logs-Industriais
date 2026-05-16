# 🏭 Industrial Machine Logs: Machine Learning & Gemini AI

Projeto Completo de Ciência de Dados e Inteligência Artificial aplicado ao setor industrial. O objetivo é analisar logs de funcionamento de máquinas pesadas, tratar anomalias, construir um modelo preditivo de Machine Learning otimizado e integrar a API do Google AI Studio (Gemini) para uma análise diagnóstica e geração de relatórios automatizados.

---

## 📊 Sobre o Projeto

Este projeto utiliza dados reais/simulados de logs de maquinários industriais para prever comportamentos e identificar falhas. A grande inovação aqui é a união do pipeline tradicional de Machine Learning com a IA Generativa (Gemini), permitindo uma interpretação muito mais rica e automatizada dos resultados obtidos.

O projeto passa por todo o ciclo de um cientista de dados:
* Engenharia de Atributos (*Feature Engineering*).
* Tratamento e visualização de outliers.
* Treinamento e validação cruzada do modelo.
* Geração de relatórios com IA através de chamadas de API.

## 🎯 Objetivos

* Transformar e limpar dados brutos de logs industriais.
* Identificar outliers e fatores de risco operacionais por meio de gráficos.
* Treinar um modelo de Machine Learning robusto.
* Evitar *overfitting* utilizando técnicas avançadas de Validação Cruzada (*Cross-Validation*).
* Integrar a API do Gemini para enriquecer a análise de dados e automatizar relatórios técnicos.

## 🛠️ Tecnologias Utilizadas

* **Linguagem Principal:** Python
* **Manipulação de Dados:** Pandas & NumPy
* **Machine Learning:** Scikit-Learn (Sklearn)
* **Visualização de Dados:** Matplotlib & Streamlit
* **Inteligência Artificial:** Google AI Studio API (Gemini Pro)

## 📁 Estrutura do Projeto

```text
📦 Industrial-Logs-ML-Gemini
┣ 📜 README.md
┣ 📜 LICENSE
┣ 📶 dataset.csv
┣ 🤖 diagnostico.txt
┗ 🐍 main.py
