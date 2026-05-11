# Predição de Níveis de Obesidade com Machine Learning

**FIAP Pós-Tech Data Analytics | Tech Challenge Fase 4**

---

## Sobre o Projeto

Sistema preditivo desenvolvido para auxiliar equipes médicas na classificação do nível de obesidade de pacientes com base em dados demográficos, hábitos alimentares e de atividade física.

**Dataset:** 2.111 pacientes | 17 variáveis | 7 classes de obesidade  
**Modelo:** Random Forest | Acurácia: 98.11% | F1-Score: 98.06%

## Estrutura do Repositório

```
├── app.py                    # Aplicação Streamlit
├── modelo_final.pkl          # Modelo treinado
├── scaler.pkl                # Scaler para normalização
├── feature_metadata.json     # Metadados do pipeline
├── requirements.txt          # Dependências
├── TC4_Obesity.ipynb  # Notebook completo
└── Obesity.csv               # Dataset original
```

## Pipeline de Machine Learning

1. **EDA** — análise exploratória com 5 visualizações
2. **Feature Engineering** — encoding, IMC derivado, normalização, split 80/20
3. **Modelagem** — Logistic Regression, Random Forest e Gradient Boosting com validação cruzada 5-fold e GridSearchCV
4. **Deploy** — aplicação interativa no Streamlit Cloud

## Links

- **Aplicação Streamlit:** [link após deploy]
- **Dashboard Analítico:** [link após criação]

## Como Rodar Localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```
