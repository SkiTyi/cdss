"""
Helpers for resolving an LLMAssistant into runtime config + calling chat APIs.

The rest of the codebase shouldn't need to care whether a given LLM is
local (vllm) or remote (OpenAI-compatible) — both expose
`{base_url, model_name, api_key}` after resolution.
"""
import json
import re
from typing import Optional

import httpx


def is_local_endpoint(url: str) -> bool:
    if not url:
        return False
    return any(h in url for h in ("localhost", "127.0.0.1", "0.0.0.0", "::1"))


def resolve_assistant(assistant) -> dict:
    """Return {base_url, model_name, api_key} for use, or raise ValueError."""
    if assistant is None:
        raise ValueError("助手不存在")
    base_url = (assistant.base_url or "").strip()
    model_name = (assistant.model_name or "").strip()
    api_key = (assistant.api_key or "").strip()

    if assistant.type == "local":
        if assistant.status != "running" or not base_url:
            raise ValueError(f"本地助手「{assistant.name}」未在运行（当前状态：{assistant.status}）")
        # vllm with --enable-lora exposes the LoRA-applied variant under
        # `<served_name>-lora` while the bare `<served_name>` is the unmodified
        # base model. For our local LoRA assistants we want callers to
        # actually exercise the fine-tuned weights — auto-suffix here so
        # downstream chat_completion sends the LoRA name.
        # See vllm_manager.start(): we register --lora-modules with this exact
        # `<served_name>-lora` key.
        if assistant.lora_adapter_path:
            model_name = f"{model_name}-lora"
    else:  # remote
        if not base_url:
            raise ValueError(f"远程助手「{assistant.name}」未配置 base_url")
        if not is_local_endpoint(base_url) and not api_key:
            raise ValueError(f"远程助手「{assistant.name}」未配置 api_key")

    if not model_name:
        raise ValueError(f"助手「{assistant.name}」未配置 model_name")

    return {"base_url": base_url, "model_name": model_name, "api_key": api_key}


def chat_completion(*, base_url: str, model_name: str, api_key: str,
                    prompt: str,
                    temperature: float = 0.3,
                    max_tokens: int = 2000,
                    timeout: float = 120.0) -> str:
    """OpenAI-compatible chat call. Returns raw assistant text content."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{base_url.rstrip('/')}/chat/completions",
                           headers=headers, json=payload)
        resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)


def parse_judge_json(text: str) -> Optional[dict]:
    """Try hard to extract a {score, reasoning} JSON object from judge output."""
    if not text:
        return None
    candidates = []
    # Code fence first
    m = _JSON_FENCE.search(text)
    if m:
        candidates.append(m.group(1))
    candidates.append(text.strip())
    # Last-resort: find first { ... } block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0))

    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict) and "score" in obj:
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return None
