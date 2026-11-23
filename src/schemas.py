from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SimNao = Literal["Yes", "No"]
ServicoInternet = Literal["Yes", "No", "No internet service"]


class ClienteEntrada(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male"] = Field(
        description="Gênero do cliente.", examples=["Female"]
    )
    SeniorCitizen: SimNao = Field(
        description="Indica se o cliente é idoso.", examples=["No"]
    )
    Partner: SimNao = Field(
        description="Indica se o cliente possui parceiro(a).", examples=["Yes"]
    )
    Dependents: SimNao = Field(
        description="Indica se o cliente possui dependentes.", examples=["No"]
    )
    tenure: int = Field(
        ge=0,
        le=120,
        description="Meses de permanência como cliente (0 a 120).",
        examples=[12],
    )
    PhoneService: SimNao = Field(
        description="Indica se o cliente possui telefonia.", examples=["Yes"]
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        description="Tipo de serviço de internet contratado.", examples=["Fiber optic"]
    )
    OnlineSecurity: ServicoInternet = Field(
        description="Serviço de segurança online.", examples=["No"]
    )
    TechSupport: ServicoInternet = Field(
        description="Serviço de suporte técnico.", examples=["No"]
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        description="Tipo de contrato.", examples=["Month-to-month"]
    )
    PaperlessBilling: SimNao = Field(
        description="Indica se a fatura é digital.", examples=["Yes"]
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(description="Forma de pagamento.", examples=["Electronic check"])
    MonthlyCharges: float = Field(
        ge=0.0,
        le=500.0,
        description="Valor cobrado mensalmente em dólares (0 a 500).",
        examples=[89.9],
    )
    TotalCharges: float = Field(
        ge=0.0,
        le=20000.0,
        description="Valor total cobrado ao longo do relacionamento em dólares (0 a 20000).",
        examples=[1078.8],
    )


class PredicaoSaida(BaseModel):
    categoria_prevista: str = Field(
        description="Classe inferida pelo modelo (Cancelou ou Permaneceu).",
        examples=["Cancelou"],
    )
    score_confianca: float = Field(
        ge=0.0,
        le=1.0,
        description="Probabilidade (predict_proba) atribuída à classe prevista.",
        examples=[0.87],
    )


class DiagnosticoSaida(BaseModel):
    status: str
    framework: str
    modelo: str
