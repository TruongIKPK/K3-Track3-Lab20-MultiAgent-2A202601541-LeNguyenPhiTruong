"""Tracing and observability hooks with Langfuse support."""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

_langfuse_client: Any = None


def get_langfuse_client() -> Any:
    """Return initialized Langfuse client or None if unconfigured."""
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client

    settings = get_settings()
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        try:
            from langfuse import Langfuse

            os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
            os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
            os.environ["LANGFUSE_HOST"] = settings.langfuse_host

            _langfuse_client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            logger.info("Langfuse tracing initialized with host: %s", settings.langfuse_host)
        except Exception as exc:
            logger.warning("Failed to initialize Langfuse client: %s", exc)
            _langfuse_client = None
    return _langfuse_client


def flush_traces() -> None:
    """Flush pending telemetry to Langfuse."""
    client = get_langfuse_client()
    if client:
        try:
            client.flush()
        except Exception as exc:
            logger.warning("Failed to flush Langfuse traces: %s", exc)


@contextmanager
def trace_span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Span context manager integrated with Langfuse and local timing."""
    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "duration_seconds": None,
    }

    client = get_langfuse_client()
    lf_context = None

    if client:
        try:
            lf_context = client.start_as_current_observation(
                name=name,
                input=attributes,
            )
            lf_context.__enter__()
        except Exception as exc:
            logger.debug("Langfuse span start failed: %s", exc)
            lf_context = None

    try:
        yield span
    except Exception as exc:
        span["error"] = str(exc)
        if lf_context:
            with suppress(Exception):
                lf_context.update(level="ERROR", status_message=str(exc))
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
        if lf_context:
            try:
                lf_context.update(
                    output={"duration_seconds": span["duration_seconds"]},
                    metadata=span["attributes"],
                )
                lf_context.__exit__(None, None, None)
            except Exception as exc:
                logger.debug("Langfuse span end failed: %s", exc)

