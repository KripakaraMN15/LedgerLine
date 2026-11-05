from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class UnknownModelPricing(ValueError):
    pass


@dataclass(frozen=True)
class ModelPricing:
    input_per_1m_tokens: float
    output_per_1m_tokens: float

    def __post_init__(self) -> None:
        if self.input_per_1m_tokens < 0 or self.output_per_1m_tokens < 0:
            raise ValueError("Token prices must be non-negative")

    def calculate(self, prompt_tokens: int, completion_tokens: int) -> float:
        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError("Token counts must be non-negative")
        return (
            prompt_tokens * self.input_per_1m_tokens
            + completion_tokens * self.output_per_1m_tokens
        ) / 1_000_000


@dataclass(frozen=True)
class PriceTable:
    models: Mapping[str, ModelPricing]

    def pricing_for(self, model: str) -> ModelPricing:
        try:
            return self.models[model]
        except KeyError as error:
            raise UnknownModelPricing(f"No token pricing configured for model {model!r}") from error

    def estimate(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        return self.pricing_for(model).calculate(prompt_tokens, completion_tokens)

    def calculate(self, model: str, usage: Mapping[str, Any]) -> float:
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if (
            not isinstance(prompt_tokens, int)
            or isinstance(prompt_tokens, bool)
            or not isinstance(completion_tokens, int)
            or isinstance(completion_tokens, bool)
        ):
            raise ValueError("Usage must include integer prompt_tokens and completion_tokens")
        return self.estimate(model, prompt_tokens, completion_tokens)


def load_price_table(path: Path | None = None) -> PriceTable:
    price_file = path or Path(__file__).resolve().parents[3] / "config" / "prices.yaml"
    with price_file.open("r", encoding="utf-8") as file:
        raw: Any = yaml.safe_load(file)
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), dict):
        raise ValueError(f"Price file {price_file} must contain a models mapping")

    models: dict[str, ModelPricing] = {}
    for model, config in raw["models"].items():
        if not isinstance(model, str) or not isinstance(config, dict):
            raise ValueError(f"Invalid model pricing entry in {price_file}")
        input_price = config.get("input_per_1m_tokens")
        output_price = config.get("output_per_1m_tokens")
        if (
            not isinstance(input_price, (int, float))
            or isinstance(input_price, bool)
            or not isinstance(output_price, (int, float))
            or isinstance(output_price, bool)
        ):
            raise ValueError(f"Invalid token prices for model {model!r}")
        models[model] = ModelPricing(float(input_price), float(output_price))
    return PriceTable(models)
