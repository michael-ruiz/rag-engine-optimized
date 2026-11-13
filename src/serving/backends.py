"""Inference backends (strategy pattern) and hardware-aware backend selection."""
from __future__ import annotations

import json
import logging
import sys
import urllib.error
import urllib.request
from typing import Any, Protocol

import torch

from src.utils.device import DeviceManager

logger = logging.getLogger(__name__)

DEFAULT_VLLM_URL = "http://127.0.0.1:8000/v1"


Messages = list[dict[str, str]]


class Backend(Protocol):
    name: str

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str: ...

    def chat(self, messages: Messages, max_new_tokens: int, temperature: float, enable_thinking: bool = False) -> str:
        """Apply the model's chat template; returned text may contain <think>...</think>."""
        ...


class VLLMBackend:
    """vLLM in-process on Linux CUDA; elsewhere, an OpenAI-compatible server (e.g. `vllm serve` via vllm-metal)."""

    def __init__(self, model_path: str, base_url: str = DEFAULT_VLLM_URL) -> None:
        self.model_path = model_path
        self.base_url = base_url.rstrip("/")
        self.native = sys.platform.startswith("linux") and torch.cuda.is_available()
        if self.native:
            from vllm import LLM

            self.llm = LLM(model=model_path, dtype="bfloat16")
            self.name = "vllm"
        else:
            self.name = "vllm-http"
            logger.info("Using vLLM server at %s (model '%s')", self.base_url, model_path)

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        if self.native:
            from vllm import SamplingParams

            params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature)
            return self.llm.generate([prompt], params)[0].outputs[0].text
        body = {"model": self.model_path, "prompt": prompt, "max_tokens": max_new_tokens, "temperature": temperature}
        return self._post("/completions", body)["choices"][0]["text"]

    def chat(self, messages: Messages, max_new_tokens: int, temperature: float, enable_thinking: bool = False) -> str:
        template_kwargs = {"enable_thinking": enable_thinking}
        if self.native:
            from vllm import SamplingParams

            params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature)
            return self.llm.chat(messages, params, chat_template_kwargs=template_kwargs)[0].outputs[0].text
        body = {
            "model": self.model_path, "messages": messages, "max_tokens": max_new_tokens,
            "temperature": temperature, "chat_template_kwargs": template_kwargs,
        }
        msg = self._post("/chat/completions", body)["choices"][0]["message"]
        # A server started with --reasoning-parser splits the trace out; re-wrap it so callers see one format.
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        return f"<think>{reasoning}</think>{msg.get('content') or ''}" if reasoning else msg.get("content") or ""

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            self.base_url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.load(resp)
        except urllib.error.URLError as e:
            raise RuntimeError(f"vLLM server at {self.base_url} unreachable or failed: {e}") from e


class TransformersMPSBackend:
    """In-process HF transformers on the best local device (mps, else cuda/cpu).

    Serves weight-patched checkpoints directly, without a separate server.
    """

    def __init__(self, model_path: str, device: str = "auto") -> None:
        from src.model.inspector import load_model

        self.model, self.tokenizer = load_model(model_path, device)
        self.name = f"transformers-{self.model.device.type}"

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        return self._generate(self.tokenizer(prompt, return_tensors="pt"), max_new_tokens, temperature)

    def chat(self, messages: Messages, max_new_tokens: int, temperature: float, enable_thinking: bool = False) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=True, return_tensors="pt", enable_thinking=enable_thinking
        )
        return self._generate(inputs, max_new_tokens, temperature)

    @torch.no_grad()
    def _generate(self, inputs, max_new_tokens: int, temperature: float) -> str:
        inputs = inputs.to(self.model.device)
        sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **sampling)
        # <think>/</think> are not special tokens in Qwen3.5, so they survive skip_special_tokens.
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def select_backend(model_path: str, backend: str = "auto", device: str = "auto", vllm_url: str = DEFAULT_VLLM_URL) -> Backend:
    """backend: "auto" | "vllm" | "transformers".

    auto -> native vLLM only on CUDA with vllm installed, otherwise transformers. A vllm-metal server is
    used only when requested explicitly, since it may be serving a different (unpatched) model.
    """
    if backend == "auto":
        backend = "transformers"
        if DeviceManager.detect(device).device == "cuda":
            try:
                import vllm  # noqa: F401

                backend = "vllm"
            except ImportError:
                logger.warning("CUDA found but vllm not installed; falling back to transformers.")
    if backend == "vllm":
        return VLLMBackend(model_path, vllm_url)
    if backend == "transformers":
        return TransformersMPSBackend(model_path, device)
    raise ValueError(f"Unknown backend '{backend}'; expected auto, vllm or transformers.")
