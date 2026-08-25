# Platform support

The dependency tree (`aiortc`, `onnxruntime`, `numba`, `cryptography`, Azure
Speech SDK) installs cleanly on:

| Platform | Notes |
|---|---|
| Linux x86_64 / arm64 | No caveats. Recommended for deployment. |
| macOS arm64 (Apple Silicon) | No caveats. |
| Windows x86_64 | No caveats. `requirements-windows.lock` is a full pinned resolution. |
| macOS x86_64 (Intel) | Works, but needs `requirements-intel-mac.override` — see below. |

## Intel macOS

Recent releases of several compiled dependencies no longer ship x86_64 macOS
wheels. Pipecat 1.7.0 pins versions on the far side of that line, so a plain
install fails to resolve. The override file pins the newest versions that still
have Intel wheels:

| Package | pipecat 1.7.0 wants | Newest x86_64 mac wheel |
|---|---|---|
| onnxruntime | ~=1.24.3 | 1.23.2 |
| numba | 0.67.0 | 0.62.1 |
| llvmlite | 0.49 | 0.45.1 |
| cryptography | 50.0.0 | 46.0.3 |

```bash
uv pip install -r requirements.txt --override requirements-intel-mac.override
```

Verified with the overrides in place: both bundled ONNX models (Silero VAD and
the smart-turn v3 model) load, the full pipeline assembles, and the server
serves. Do not use the overrides on any other platform — they are a downgrade.

## HTTPS certificate errors

The python.org macOS installer ships without a wired-up CA bundle, so stdlib
HTTPS can fail with `CERTIFICATE_VERIFY_FAILED`. Fix either way:

```bash
open "/Applications/Python 3.11/Install Certificates.command"  # system-wide
pip install certifi                                            # per-env
```

If you are behind a TLS-intercepting corporate proxy, export a bundle that
includes your proxy's root certificate and point `SSL_CERT_FILE` at it.

## Browser requirements

Microphone access needs a secure context: `localhost` qualifies, any other
host needs HTTPS. The browser must also be able to reach the server's ICE
candidates directly — WebRTC media negotiates its own UDP path, so an SSH
tunnel that only forwards the HTTP port is not enough.
