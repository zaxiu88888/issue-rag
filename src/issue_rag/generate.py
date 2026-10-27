"""Generation backends.

Two implementations, one interface: a local HuggingFace chat model (the default,
because the whole point is that anyone can reproduce the numbers with no API
key) and an OpenAI-compatible endpoint for people who would rather not download
weights.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


class ChatModel(Protocol):
    name: str

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> GenerationResult: ...


class LocalChatModel:
    """A causal LM loaded through transformers, generating greedily by default."""

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        max_context_tokens: int = 8_192,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_path
        self.torch = torch
        self.max_context_tokens = max_context_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        torch_dtype = getattr(torch, dtype, torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def _render(self, messages: list[dict[str, str]]) -> str:
        """Use the model's own chat template so prompts match its training format."""
        apply = getattr(self.tokenizer, "apply_chat_template", None)
        if apply is None:
            parts = [f"{m['role'].upper()}: {m['content']}" for m in messages]
            return "\n\n".join(parts) + "\n\nASSISTANT:"
        return apply(messages, tokenize=False, add_generation_prompt=True)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> GenerationResult:
        torch = self.torch
        prompt = self._render(messages)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_context_tokens
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[1])

        do_sample = temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        started = time.time()
        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)
        latency = time.time() - started

        generated = output[0][prompt_tokens:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=int(generated.shape[0]),
            latency_s=latency,
        )


class OpenAICompatChatModel:
    """Any ``/v1/chat/completions`` endpoint (vLLM, Ollama, LM Studio, OpenAI)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "not-needed",
        timeout_s: int = 120,
    ) -> None:
        self.name = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> GenerationResult:
        import json
        import urllib.request

        payload = json.dumps(
            {
                "model": self.name,
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        started = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = json.loads(resp.read())
        latency = time.time() - started
        usage = body.get("usage") or {}
        return GenerationResult(
            text=(body["choices"][0]["message"]["content"] or "").strip(),
            prompt_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_s=latency,
        )


# --------------------------------------------------------------------- provider


class MissingCredential(RuntimeError):
    """The chosen provider needs a key that is not in the environment."""


# Shorthands so the common cases are one flag instead of three. Anything
# OpenAI-compatible can also be addressed directly with --endpoint <url>.
PROVIDERS: dict[str, dict[str, str]] = {
    "local": {},
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "key_env": "SILICONFLOW_API_KEY",
    },
    "ollama": {
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen2.5:7b",
        "key_env": "OLLAMA_API_KEY",
    },
}


def resolve_model(
    *,
    provider: str = "local",
    endpoint: str = "",
    model: str = "",
    local_default: str = "data/models/Qwen2.5-3B-Instruct",
) -> ChatModel:
    """Build the chat model from CLI flags plus whatever is in ``.env``.

    Precedence: an explicit ``--endpoint`` URL always wins; otherwise the named
    provider's base URL is used; ``--model`` overrides the provider default so
    you can point at a different model on the same service.
    """
    import os

    if provider not in PROVIDERS:
        raise SystemExit(
            f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)} "
            "or pass --endpoint <url>"
        )
    spec = PROVIDERS[provider]

    # A bare --endpoint pointing at a local server, with no provider chosen.
    if endpoint and provider == "local":
        return OpenAICompatChatModel(
            endpoint, model or local_default, os.environ.get("OPENAI_API_KEY", "not-needed")
        )

    if provider == "local" and not endpoint:
        return LocalChatModel(model or local_default)

    base_url = endpoint or spec.get("base_url", "")
    served_model = model or spec.get("model", "")
    key_env = spec.get("key_env", "")
    api_key = os.environ.get(key_env, "") if key_env else "not-needed"

    if key_env and not api_key:
        raise MissingCredential(
            f"provider {provider!r} needs {key_env}.\n"
            f"Put it in {os.path.join(os.getcwd(), '.env')} as:\n"
            f"    {key_env}=sk-...\n"
            "or export it in the shell that runs the command."
        )
    return OpenAICompatChatModel(base_url, served_model, api_key or "not-needed")
