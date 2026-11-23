# AT2 - Redes Neurais MLP e Deploy de API em Produção

Previsão de cancelamento de clientes (*churn*) de uma operadora de telecomunicações com uma rede neural multicamadas (`MLPClassifier`), encapsulada em um `Pipeline` do Scikit-Learn e servida por uma API REST com FastAPI.

## Dataset

- **Telco Customer Churn** (Kaggle): <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>
- Arquivo original: `WA_Fn-UseC_-Telco-Customer-Churn.csv`, salvo em `data/dataset.csv` (7.043 registros, 21 colunas).
- Problema: classificação binária (`Cancelou` x `Permaneceu`), com variáveis numéricas e categóricas.

## Resultados (conjunto de teste, 20% estratificado, `random_state=77`)

| Métrica | Valor |
|---|---|
| Acurácia | 0,813 |
| MCC (Matthews) | 0,490 |

Arquitetura: `hidden_layer_sizes=(128, 64, 32)`, `learning_rate_init=0.001`, `alpha=0.001`, `early_stopping=True`, `random_state=77`.

## Estrutura

```
.
├── data/dataset.csv
├── models/pipeline_rede_neural.joblib
├── notebooks/analise_treinamento.ipynb
├── src/
│   ├── __init__.py
│   ├── schemas.py
│   └── app.py
├── pyproject.toml
├── requisicoes.http
└── README.md
```

## Como reproduzir

Pré-requisitos: [uv](https://docs.astral.sh/uv/) e Python 3.11 ou superior.

```bash
uv sync
```

### 1. Treinar o modelo (opcional, o `.joblib` já acompanha o repositório)

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/analise_treinamento.ipynb
```

Ou abra `notebooks/analise_treinamento.ipynb` no Jupyter/VS Code e execute todas as células. O pipeline completo é salvo em `models/pipeline_rede_neural.joblib`.

### 2. Subir a API

```bash
uv run uvicorn src.app:app --host 127.0.0.1 --port 8000
```

Documentação interativa: <http://127.0.0.1:8000/docs>

### 3. Testar

Abra `requisicoes.http` no VS Code (extensão REST Client) ou em outra IDE compatível e execute as requisições. Todas enviam o cabeçalho `User-Agent: Tester-Audit-Agent`.

| Requisição | Resultado esperado |
|---|---|
| `GET /diagnostico/modelo` | 200 com `status`, `framework` e `modelo` |
| `POST /predict` (perfil A) | 200, `Cancelou` |
| `POST /predict` (perfil B) | 200, `Permaneceu` |
| `POST /predict` (texto em campo numérico) | 422 |
| `POST /predict` (valores fora dos limites) | 422 |

## Endpoints

### `GET /diagnostico/modelo`

```json
{
  "status": "operacional_v2",
  "framework": "FastAPI-Production",
  "modelo": "MLPClassifier"
}
```

### `POST /predict`

Entrada (todos os campos obrigatórios; campos desconhecidos são rejeitados):

```json
{
  "gender": "Female",
  "SeniorCitizen": "No",
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "TechSupport": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 95.7,
  "TotalCharges": 191.4
}
```

Saída:

```json
{
  "categoria_prevista": "Cancelou",
  "score_confianca": 0.74
}
```

`score_confianca` é a probabilidade (`predict_proba`) da classe prevista, entre 0.0 e 1.0.

## Notas de engenharia

- O pipeline é carregado uma única vez no `lifespan` da aplicação e guardado em `app.state.modelo`; nenhuma rota lê o disco.
- O `.joblib` é desserializado com `joblib.load`, que executa código arbitrário se o arquivo vier de fonte não confiável. Use apenas o arquivo gerado por este projeto.
