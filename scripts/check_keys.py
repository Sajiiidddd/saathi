#!/usr/bin/env python3
"""Validate credentials before wasting a call on them.

    python3 scripts/check_keys.py

Reads .env (no dotenv dependency — stdlib only, so this runs before or without
the venv). Checks three things that each fail in a confusingly silent way at
runtime:

  1. Azure key + region actually authenticate      -> otherwise: no audio, no error
  2. The configured TTS voice exists in THAT region -> otherwise: silence
  3. Gemini key works and the model id is real      -> otherwise: 404 mid-turn

Exit code 0 = good to go.
"""

from __future__ import annotations

import json
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
OK, FAIL, WARN = f"{GREEN}✓{RESET}", f"{RED}✗{RESET}", f"{YELLOW}!{RESET}"


def read_env() -> dict[str, str]:
    path = ROOT / ".env"
    if not path.exists():
        print(f"{FAIL} No .env found at {path}")
        print(f"  {DIM}cp .env.example .env{RESET}  then fill in your keys")
        sys.exit(1)
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip("'\"")
    return env


class CertsMissing(RuntimeError):
    """The interpreter has no usable CA bundle."""


def _ssl_context() -> ssl.SSLContext:
    """Build a verifying context that survives a bare python.org install.

    Resolution order:
    1. An explicit SSL_CERT_FILE that actually exists.
    2. The repo's certs/ca-bundle.pem, if present — on a machine behind a
       TLS-intercepting proxy this is the bundle carrying the proxy root,
       and loading it explicitly beats trusting env-var plumbing.
    3. certifi, if importable.
    4. The interpreter default (which on a bare python.org macOS install may
       verify nothing — hence the CertsMissing remedy).
    Verification is never disabled — these requests carry API keys.
    """
    import os

    explicit = os.getenv("SSL_CERT_FILE")
    if explicit and Path(explicit).exists():
        return ssl.create_default_context(cafile=explicit)
    repo_bundle = ROOT / "certs" / "ca-bundle.pem"
    if repo_bundle.exists():
        return ssl.create_default_context(cafile=str(repo_bundle))
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


CTX = _ssl_context()


def _open(req: urllib.request.Request, timeout: int) -> bytes:
    # Cloudflare-fronted APIs (Groq among them) 403 requests with no
    # User-Agent, which reads as a mystery auth failure.
    if not req.has_header("User-agent"):
        req.add_header("User-Agent", "Saathi-check/1.0")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=CTX) as resp:
            return resp.read()
    except urllib.error.URLError as exc:
        if isinstance(getattr(exc, "reason", None), ssl.SSLCertVerificationError):
            raise CertsMissing(
                "This Python has no CA certificates, so it cannot verify HTTPS.\n"
                "  Fix either way:\n"
                "    open '/Applications/Python 3.11/Install Certificates.command'\n"
                "  or, without touching system Python:\n"
                "    pip install certifi   (then re-run this script)"
            ) from exc
        raise


def post(url: str, headers: dict[str, str]) -> bytes:
    req = urllib.request.Request(url, data=b"", headers=headers, method="POST")
    return _open(req, 20)


def get(url: str, headers: dict[str, str] | None = None) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})
    return _open(req, 30)


def check_azure(env: dict[str, str]) -> bool:
    key = env.get("AZURE_SPEECH_API_KEY", "")
    region = env.get("AZURE_SPEECH_REGION", "")
    voice = env.get("SAATHI_TTS_VOICE") or "en-IN-NeerjaNeural"

    if not key or not region:
        print(f"{FAIL} Azure: AZURE_SPEECH_API_KEY / AZURE_SPEECH_REGION not filled in")
        return False

    if region != region.lower() or " " in region:
        print(f"{FAIL} Azure region '{region}' looks like a display name.")
        print(f"  {DIM}Use the slug from your resource's Endpoint URL, e.g. 'centralindia'{RESET}")
        return False

    # 1. Does the key authenticate in this region?
    try:
        post(
            f"https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken",
            {"Ocp-Apim-Subscription-Key": key, "Content-Length": "0"},
        )
        print(f"{OK} Azure key authenticates in region '{region}'")
    except urllib.error.HTTPError as exc:
        print(f"{FAIL} Azure auth failed: HTTP {exc.code}")
        if exc.code == 401:
            print(f"  {DIM}Wrong key, or the key belongs to a different region{RESET}")
        elif exc.code == 404:
            print(f"  {DIM}Region '{region}' has no Speech endpoint — check the slug{RESET}")
        return False
    except urllib.error.URLError as exc:
        print(f"{FAIL} Could not reach Azure ({exc.reason}) — network or proxy?")
        return False

    # 2. Does the configured voice exist HERE? Voice availability is per-region.
    try:
        raw = get(
            f"https://{region}.tts.speech.microsoft.com/cognitiveservices/voices/list",
            {"Ocp-Apim-Subscription-Key": key},
        )
        voices = json.loads(raw)
        names = {v.get("ShortName") for v in voices}
        if voice in names:
            match = next(v for v in voices if v.get("ShortName") == voice)
            print(
                f"{OK} Voice '{voice}' available "
                f"{DIM}({match.get('LocalName','?')}, {match.get('Gender','?')}){RESET}"
            )
        else:
            print(f"{FAIL} Voice '{voice}' NOT available in '{region}'")
            indian = sorted(n for n in names if n.startswith(("en-IN", "hi-IN")))
            if indian:
                print(f"  {DIM}en-IN / hi-IN voices here: {', '.join(indian[:10])}{RESET}")
            print(f"  {DIM}Set SAATHI_TTS_VOICE in .env to one of those{RESET}")
            return False
        print(f"  {DIM}{len(names)} voices total in this region{RESET}")
    except urllib.error.HTTPError as exc:
        print(f"{WARN} Could not list voices (HTTP {exc.code}) — auth worked, so probably fine")

    return True


def check_gemini(env: dict[str, str]) -> bool:
    key = env.get("GOOGLE_API_KEY", "")
    wanted = env.get("SAATHI_LLM_MODEL") or "gemini-2.5-flash"

    if not key:
        print(f"{FAIL} Gemini: GOOGLE_API_KEY not filled in")
        return False

    try:
        raw = get(f"https://generativelanguage.googleapis.com/v1beta/models?key={key}")
        models = json.loads(raw).get("models", [])
        ids = {m["name"].removeprefix("models/") for m in models}
        print(f"{OK} Gemini key works {DIM}({len(ids)} models visible){RESET}")

        if wanted in ids:
            print(f"{OK} Model '{wanted}' available")
        else:
            print(f"{FAIL} Model '{wanted}' not in your model list")
            flash = sorted(m for m in ids if "flash" in m and "image" not in m)
            print(f"  {DIM}flash models you can use: {', '.join(flash[:8])}{RESET}")
            print(f"  {DIM}Set SAATHI_LLM_MODEL in .env{RESET}")
            return False
        return True
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore")[:200]
        print(f"{FAIL} Gemini check failed: HTTP {exc.code}")
        if exc.code in (400, 403):
            print(f"  {DIM}Bad key, or AI Studio is blocked for this Google account{RESET}")
            print(f"  {DIM}(Workspace/corporate accounts are often blocked — use a personal Gmail){RESET}")
        print(f"  {DIM}{body}{RESET}")
        return False
    except urllib.error.URLError as exc:
        print(f"{FAIL} Could not reach Google ({exc.reason}) — network or proxy?")
        return False


def check_azure_openai(env: dict[str, str]) -> bool:
    endpoint = env.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    key = env.get("AZURE_OPENAI_API_KEY", "")
    deployment = env.get("AZURE_OPENAI_DEPLOYMENT", "")
    if not (endpoint and key and deployment):
        print(f"{FAIL} Azure OpenAI: fill AZURE_OPENAI_ENDPOINT, _API_KEY and _DEPLOYMENT")
        return False
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=2024-10-21"
    body = json.dumps(
        {"messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 5}
    ).encode()
    request = urllib.request.Request(
        url, data=body, headers={"api-key": key, "Content-Type": "application/json"}
    )
    try:
        data = json.loads(_open(request, 30))
        reply = data["choices"][0]["message"]["content"]
        print(f"{OK} Azure OpenAI deployment '{deployment}' answers: {reply!r}")
        return True
    except urllib.error.HTTPError as exc:
        print(f"{FAIL} Azure OpenAI: HTTP {exc.code}")
        if exc.code == 404:
            print(f"  {DIM}No deployment named '{deployment}' on that resource — the value "
                  f"must be YOUR deployment name from AI Foundry, not the model id{RESET}")
        elif exc.code == 401:
            print(f"  {DIM}Key doesn't match this resource{RESET}")
        return False


def check_openai_compat(env: dict[str, str]) -> bool:
    key = env.get("OPENAI_API_KEY", "")
    base = (env.get("OPENAI_BASE_URL", "") or "https://api.openai.com/v1").rstrip("/")
    model = env.get("OPENAI_MODEL", "openai/gpt-4o-mini")
    if not key:
        print(f"{FAIL} OPENAI_API_KEY not set (for GitHub Models: a fine-grained PAT "
              f"with the 'Models' permission)")
        return False
    body = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 5}
    ).encode()
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        data = json.loads(_open(request, 30))
        reply = data["choices"][0]["message"]["content"]
        print(f"{OK} OpenAI-compatible endpoint answers via {base}")
        print(f"{OK} Model '{model}' replies: {reply!r}")
        return True
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")[:160]
        print(f"{FAIL} {base}: HTTP {exc.code}")
        if exc.code == 401:
            print(f"  {DIM}Token rejected — for GitHub Models the PAT needs the "
                  f"'Models' account permission{RESET}")
        elif exc.code == 404:
            print(f"  {DIM}Model id '{model}' not found on this endpoint — GitHub Models "
                  f"ids look like 'openai/gpt-4o-mini'{RESET}")
        print(f"  {DIM}{detail}{RESET}")
        return False


def main() -> int:
    env = read_env()
    provider = (env.get("SAATHI_LLM_PROVIDER") or "gemini").lower()
    try:
        print("\nChecking Azure Speech…")
        azure_ok = check_azure(env)
        if provider == "azure":
            print("\nChecking Azure OpenAI…")
            gemini_ok = check_azure_openai(env)
        elif provider == "openai":
            print("\nChecking OpenAI-compatible endpoint…")
            gemini_ok = check_openai_compat(env)
        else:
            print("\nChecking Gemini…")
            gemini_ok = check_gemini(env)
    except CertsMissing as exc:
        print(f"\n{FAIL} {exc}\n")
        return 1

    print()
    if azure_ok and gemini_ok:
        print(f"{GREEN}Both credentials good — start the server:{RESET}")
        print(f"  {DIM}.venv/bin/python server.py{RESET}\n")
        return 0
    print(f"{RED}Fix the above before running server.py.{RESET}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
