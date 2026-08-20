"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
from dataclasses import dataclass

from openai import OpenAI

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client with OpenAI backend and safe fallback."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        self._client: OpenAI | None = OpenAI(api_key=self.api_key) if self.api_key else None

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Return a model completion."""
        if not self._client:
            logger.warning("OPENAI_API_KEY not configured. Returning fallback response.")
            return LLMResponse(
                content=f"[Fallback response for prompt]\n{user_prompt[:300]}",
                input_tokens=100,
                output_tokens=100,
                cost_usd=0.0,
            )

        with trace_span(
            "llm_completion",
            {"model": self.model, "system_prompt": system_prompt[:100], "temperature": temperature},
        ) as span:
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                )
                content = response.choices[0].message.content or ""
                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else None
                output_tokens = usage.completion_tokens if usage else None

                # Approximate cost for gpt-4o-mini ($0.15/1M input, $0.60/1M output)
                cost: float | None = None
                if input_tokens is not None and output_tokens is not None:
                    cost = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000

                span["attributes"].update(
                    {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost_usd": cost,
                    }
                )

                return LLMResponse(
                    content=content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                )
            except Exception as exc:
                logger.error("LLM completion failed: %s", exc)
                raise exc
