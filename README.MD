# Customer Churn Intelligence

## Visao Geral
Este projeto implementa uma solucao de analise e previsao de churn para clientes de telecom, combinando:

- Analise exploratoria de dados (EDA)
- Modelo supervisionado para estimar probabilidade de churn
- Regras de negocio orientadas a retencao
- Dashboard interativo em Streamlit para apoio a decisao

A aplicacao foi pensada para uso pratico: transformar dados historicos em sinais acionaveis para reduzir cancelamentos.

## Objetivo de Negocio
O foco da solucao e identificar clientes com risco de churn de forma antecipada, priorizando **recall** (capturar o maximo de churners possivel) para habilitar estrategias de retencao.

Regra central adotada:

- Threshold operacional de risco: `0.15`
- Custo de falso negativo maior que falso positivo (FN:FP = 5:1)
- Meta operacional de recall: `75%`

## Como a Solucao Funciona
### 1. Carregamento e preparacao dos dados
- Busca automatica do arquivo de dados no diretorio do projeto
- Tratamento de `TotalCharges` (conversao numerica e imputacao por mediana)
- Conversao de `SeniorCitizen` para categoria textual

### 2. Treinamento do modelo
- Modelo base: `LogisticRegression`
- Pipeline com:
- `StandardScaler` para variaveis numericas
- `OneHotEncoder` para variaveis categoricas
- Calibracao de probabilidade com `CalibratedClassifierCV` (metodo `sigmoid`)

### 3. Avaliacao e interpretabilidade
- Matriz de confusao no threshold operacional
- Importancia global por coeficientes medios dos folds calibrados
- Explicabilidade local no simulador por contribuicao linear das features

### 4. Aplicacao Streamlit
A interface possui tres paginas:

1. **Overview and Analysis**
- KPIs principais
- Distribuicoes e boxplots de churn
- Graficos de drivers por segmento

2. **Model Performance and Rules**
- Resumo tecnico e de negocio do modelo
- Importancia de variaveis
- Matriz de confusao

3. **Churn Simulator**
- Formulario de cliente
- Probabilidade calibrada de churn
- Gauge de risco
- Top contribuicoes para a previsao

## Estrutura do Projeto
- `app.py`: aplicacao Streamlit (EDA, treino, avaliacao e simulador)
- `Customer_churn_pred.ipynb`: notebook de estudo e experimentacao
- `requirements.txt`: dependencias do projeto
- `WA_Fn-UseC_-Telco-Customer-Churn.xls`: base de dados
- `Plots/`: imagens e graficos auxiliares

## Como Executar
### 1. Criar e ativar ambiente virtual (opcional, recomendado)
No Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Instalar dependencias
```powershell
pip install -r requirements.txt
```

### 3. Iniciar a aplicacao
```powershell
streamlit run app.py
```

## Dependencias Principais
- `streamlit`
- `pandas`
- `numpy`
- `plotly`
- `scikit-learn`

## Limitacoes e Proximos Passos
- Validar desempenho em dados mais recentes (drift temporal)
- Testar outros algoritmos (ex.: gradient boosting) mantendo calibracao
- Evoluir estrategia de threshold por segmento de cliente
- Integrar recomendacoes de acao de retencao por perfil

## Resumo
Esta solucao entrega uma base completa de **analise + predicao + explicacao + interface**, pronta para apoiar decisoes de retencao de clientes em contexto de churn.