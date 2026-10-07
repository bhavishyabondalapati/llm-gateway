"""Cost calculation from token usage and the price table in the YAML config."""

from gateway.config import GatewayConfig


def compute_cost(config: GatewayConfig, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = config.pricing.get(model, config.default_price)
    return (
        prompt_tokens * price.input_per_million + completion_tokens * price.output_per_million
    ) / 1_000_000
