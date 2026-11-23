from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from src.schemas import ClienteEntrada, DiagnosticoSaida, PredicaoSaida

CAMINHO_MODELO = Path(__file__).resolve().parent.parent / "models" / "pipeline_rede_neural.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.modelo = joblib.load(CAMINHO_MODELO)
    yield
    app.state.modelo = None


app = FastAPI(
    title="API de Previsão de Churn (MLP)",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/diagnostico/modelo", response_model=DiagnosticoSaida)
async def diagnostico_modelo() -> DiagnosticoSaida:
    return DiagnosticoSaida(
        status="operacional_v2",
        framework="FastAPI-Production",
        modelo="MLPClassifier",
    )


@app.post("/predict", response_model=PredicaoSaida, status_code=200)
def predict(entrada: ClienteEntrada, request: Request) -> PredicaoSaida:
    modelo = getattr(request.app.state, "modelo", None)
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo indisponível.")

    quadro = pd.DataFrame([entrada.model_dump()])
    probabilidades = modelo.predict_proba(quadro)[0]
    indice = int(np.argmax(probabilidades))

    return PredicaoSaida(
        categoria_prevista=str(modelo.classes_[indice]),
        score_confianca=float(probabilidades[indice]),
    )
