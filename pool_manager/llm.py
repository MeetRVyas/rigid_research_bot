from __future__ import annotations
from typing import Any, Dict, Optional, Type, Union

from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import RateLimitError
from pydantic import BaseModel


try:
    from google.genai.errors import ClientError as GoogleGenAIClientError
except ImportError:  # pragma: no cover
    class GoogleGenAIClientError(Exception):  # type: ignore[no-redef]
        pass

from pool_manager.endpoint_state import EndpointState
from pool_manager.scheduler import AllEndpointsExhausted, LLMScheduler

from pathlib import Path
import yaml
import os


def _is_rate_limit_error(exc: BaseException) -> bool:
    """
    True if `exc` represents a 429 / rate-limit response from any provider
    client this pool might use.
    """
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, GoogleGenAIClientError):
        return getattr(exc, "code", None) == 429
    return False


def build_endpoints(config: str | Path, temperature: float = 0.0) -> list[EndpointState]:
    """
    config shape, per provider:
        {
            "models": [...],
            "keys": [...],
            "base_url": "...",        # optional
            "rpm_limit": int,         # required — requests/minute for this provider's models
            "rpd_limit": int,         # required — requests/day for this provider's models
        }
    """
    endpoints: list[EndpointState] = []

    with open(config) as f:
        config = yaml.safe_load(f)

    for provider, params in config.items():
        if "rpm_limit" not in params or "rpd_limit" not in params:
            raise ValueError(
                f"llm_config['{provider}'] is missing rpm_limit/rpd_limit — "
                "EndpointState requires both. Add the provider's published "
                "per-key rate limits."
            )

        rpm_limit = params["rpm_limit"]
        rpd_limit = params["rpd_limit"]

        for key in params["keys"]:
            api_key = os.environ.get(key)
            if not api_key:
                raise ValueError(
                    f"Environment variable '{key}' is not set"
                )
            for model in params["models"]:
                client_kwargs: Dict[str, Any] = {
                    "model": model,
                    "api_key": api_key,
                    "temperature": temperature,
                }
                if params.get("base_url"):
                    client_kwargs["base_url"] = params["base_url"]

                endpoints.append(
                    EndpointState(
                        id=f"{provider}:{model}:{key}",
                        provider=provider,
                        model=model,
                        rpm_limit=rpm_limit,
                        rpd_limit=rpd_limit,
                        client=ChatGoogleGenerativeAI(**client_kwargs) if provider.lower() in ("google", "gemini") else ChatOpenAI(**client_kwargs),
                    )
                )

    return endpoints


class _MultiProviderStructuredRunnable(Runnable[LanguageModelInput, Union[Dict, BaseModel]]):
    """
    Runnable returned by MultiProviderChatLLM.with_structured_output().

    MultiProviderChatLLM doesn't hold a single bound client — it picks a
    (provider, model, key) endpoint from the scheduler fresh on every call.
    So structured output can't be resolved once at bind time the way a
    normal single-provider ChatModel would.

    Instead, each ainvoke() here allocates an endpoint exactly like
    MultiProviderChatLLM._agenerate does, then defers to *that endpoint's*
    own `.with_structured_output(...)`. That means each provider uses its
    native mechanism under the hood (OpenAI tool calling, Gemini function
    calling / JSON mode, etc.) — this class does no schema translation of
    its own.

    Caveat: if your pool mixes providers, `**kwargs` passed here (e.g. a
    `method="json_mode"` override) must be valid for every provider in the
    pool, since the same kwargs are forwarded regardless of which endpoint
    gets picked for a given call.
    """

    def __init__(
        self,
        scheduler: LLMScheduler,
        schema: Optional[Union[Dict, Type[BaseModel]]],
        *,
        include_raw: bool = False,
        structured_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.scheduler = scheduler
        self.schema = schema
        self.include_raw = include_raw
        self.structured_kwargs = structured_kwargs or {}

    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Union[Dict, BaseModel]:
        raise NotImplementedError(
            "MultiProviderChatLLM is async-only — use ainvoke()/abatch() on "
            "the structured-output runnable, not invoke()/batch()."
        )

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Union[Dict, BaseModel]:
        while True:
            async with self.scheduler.allocate_endpoint() as ep:
                try:
                    structured_client = ep.client.with_structured_output(
                        self.schema,
                        include_raw=self.include_raw,
                        **self.structured_kwargs,
                    )
                    return await structured_client.ainvoke(input, config=config, **kwargs)

                except Exception as exc:
                    if _is_rate_limit_error(exc):
                        ep.mark_rate_limited()
                        continue
                    raise


class MultiProviderChatLLM(BaseChatModel):
    scheduler: LLMScheduler

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "multi_provider_chat"

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        while True:
            async with self.scheduler.allocate_endpoint() as ep:
                try:
                    call_kwargs = dict(kwargs)
                    if stop is not None:
                        call_kwargs["stop"] = stop

                    response = await ep.client.ainvoke(messages, **call_kwargs)

                    return ChatResult(
                        generations=[
                            ChatGeneration(
                                message=AIMessage(
                                    content=response.content,
                                    response_metadata=getattr(response, "response_metadata", {}),
                                    usage_metadata=getattr(response, "usage_metadata", None),
                                )
                            )
                        ]
                    )

                except Exception as exc:
                    if _is_rate_limit_error(exc):
                        ep.mark_rate_limited()
                        continue
                    raise

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise NotImplementedError(
            "MultiProviderChatLLM is async-only — use ainvoke()/abatch(), "
            "not invoke()/batch()."
        )

    def with_structured_output(
        self,
        schema: Optional[Union[Dict, Type[BaseModel]]] = None,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """
        Returns a Runnable that produces structured output matching `schema`
        (a Pydantic model, TypedDict, or JSON schema dict — same as the
        standard LangChain with_structured_output API).

        Only ainvoke()/abatch() are supported on the returned runnable,
        matching the rest of this class. Each call independently allocates
        an endpoint from the scheduler, so different calls (or retries
        within a call) may land on different providers/models/keys — see
        _MultiProviderStructuredRunnable for details and caveats around
        mixed-provider pools.
        """
        return _MultiProviderStructuredRunnable(
            scheduler=self.scheduler,
            schema=schema,
            include_raw=include_raw,
            structured_kwargs=kwargs,
        )


def get_llm(
    config: str | Path,
    temperature: float = 0.0,
    max_wait_seconds: float = 120.0,
) -> MultiProviderChatLLM:
    endpoints = build_endpoints(config, temperature=temperature)
    scheduler = LLMScheduler(endpoints, max_wait_seconds=max_wait_seconds)
    return MultiProviderChatLLM(scheduler=scheduler)