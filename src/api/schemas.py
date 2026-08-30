"""Contratos de entrada e saída da API.

Os nomes dos campos são idênticos aos das colunas do CSV original — assim
o payload JSON vira um ``DataFrame`` sem nenhum mapeamento intermediário.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SessionFeatures(BaseModel):
    """Features comportamentais de uma sessão de navegação."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "Administrative": 2,
                "Administrative_Duration": 80.0,
                "Informational": 0,
                "Informational_Duration": 0.0,
                "ProductRelated": 38,
                "ProductRelated_Duration": 1495.0,
                "BounceRates": 0.0,
                "ExitRates": 0.015,
                "PageValues": 32.5,
                "SpecialDay": 0.0,
                "Month": "Nov",
                "OperatingSystems": "2",
                "Browser": "2",
                "Region": "1",
                "TrafficType": "2",
                "VisitorType": "Returning_Visitor",
                "Weekend": "False",
            }
        },
    )

    administrative: int = Field(alias="Administrative", ge=0)
    administrative_duration: float = Field(alias="Administrative_Duration", ge=0)
    informational: int = Field(alias="Informational", ge=0)
    informational_duration: float = Field(alias="Informational_Duration", ge=0)
    product_related: int = Field(alias="ProductRelated", ge=0)
    product_related_duration: float = Field(alias="ProductRelated_Duration", ge=0)
    bounce_rates: float = Field(alias="BounceRates", ge=0, le=1)
    exit_rates: float = Field(alias="ExitRates", ge=0, le=1)
    page_values: float = Field(alias="PageValues", ge=0)
    special_day: float = Field(alias="SpecialDay", ge=0, le=1)
    month: str = Field(alias="Month")
    operating_systems: str = Field(alias="OperatingSystems")
    browser: str = Field(alias="Browser")
    region: str = Field(alias="Region")
    traffic_type: str = Field(alias="TrafficType")
    visitor_type: str = Field(alias="VisitorType")
    weekend: str = Field(alias="Weekend")


class PredictResponse(BaseModel):
    """Resultado da predição de propensão de compra."""

    purchase_probability: float = Field(description="Probabilidade de conversão (0 a 1).")
    will_purchase: bool = Field(description="Classe predita no limiar configurado.")
    threshold: float = Field(description="Limiar de decisão aplicado.")


class HealthResponse(BaseModel):
    """Estado do serviço e origem do modelo carregado."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_source: str | None = None
