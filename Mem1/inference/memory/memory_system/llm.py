import asyncio
import requests

from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union, Protocol
from openai import OpenAI

class LLMClient(Protocol):
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        ) -> str:
        ...

class OpenAIClient:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        client: Optional[OpenAI] = None,
        *,
        backend: str = "openai", 
        vllm_url: str = "http://localhost:8014", 
        vllm_model: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        self._backend = backend.lower()
        self._model = model
        self._timeout = timeout

        if self._backend == "openai":
            self._client = client or OpenAI()
        else:
            self._client = client

        self._vllm_url = vllm_url.rstrip("/") if vllm_url else None
        self._vllm_model = vllm_model

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.01,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        stop: Optional[list[str]] = None,
    ) -> str:
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                return await self._complete_once(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                )
            except Exception as exc:
                last_error = exc
                if attempt == max_retries:
                    raise last_error
                delay = retry_delay * (2 ** attempt)
                if delay > 0:
                    await asyncio.sleep(delay)

        raise last_error or RuntimeError("LLM completion failed.")

    async def _complete_once(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        stop: Optional[list[str]] = None,
    ) -> str:
        if self._backend == "vllm":
            return await self._complete_once_vllm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
        else:
            return await self._complete_once_openai(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    async def _complete_once_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Optional[Exception] = None

        try:
            response = await asyncio.to_thread(
                self._client.responses.create,
                model=self._model,
                input=messages,
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            if hasattr(response, "output_text"):
                return response.output_text
        except (AttributeError, TypeError) as exc:
            last_error = exc

        try:
            response = await asyncio.to_thread(
                self._client.chat.completions.create,
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            message = response.choices[0].message
            return message["content"] if isinstance(message, dict) else message.content
        except AttributeError as exc:
            last_error = last_error or exc

        try:
            response = await asyncio.to_thread(
                self._client.ChatCompletion.create,
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message["content"]
        except Exception as exc:
            raise last_error or exc


    async def _complete_once_vllm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        stop: Optional[list[str]] = None,
    ) -> str:
        if not self._vllm_url:
            raise RuntimeError(
                "vLLM backend selected but vllm_url is not set."
            )

        model_name = self._vllm_model or self._model

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        if stop:
            payload["stop"] = stop

        def _post():
            return requests.post(
                f"{self._vllm_url}/v1/chat/completions",
                json=payload,
                timeout=self._timeout,
            )

        response = await asyncio.to_thread(_post)
        status = response.status_code

        try:
            data = response.json()
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse JSON from vLLM (status={status}): "
                f"{response.text[:500]}"
            ) from e

        if status != 200:
            raise RuntimeError(
                f"vLLM returned error (status={status}): {data}"
            )

        if "choices" not in data:
            raise RuntimeError(
                f"vLLM response missing 'choices': {data}"
            )

        choice = data["choices"][0]
        message = choice.get("message", {})
        if isinstance(message, dict):
            content = (message.get("content") or "").strip()
        else:
            content = (getattr(message, "content", "") or "").strip()

        return content
