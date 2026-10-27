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
