# ModelDesk 1.2.0 verification — 2026-10-16

Source: working patch based on `c8b8e2b`; ModelDesk version 1.2.0. Python/native
inference algorithms are unchanged. Python changes add the pinned external model
profile and remove the former 32 GB validation ceiling.

## Checks completed

| Check | Result |
|---|---|
| `dotnet test studio/tests/ModelDesk.Tests/ModelDesk.Tests.csproj --no-restore -c Release` | 309 passed, zero failed/skipped |
| `.venv-reference/Scripts/python.exe -m unittest discover -s tests -v` | 927 discovered, 862 passed, 65 skipped, zero failures |
| `python -m unittest tests.test_audit tests.test_winjob -v` after Windows readback expansion | 20 passed, zero failures/skips |
| `publish-studio.ps1 -SkipTests` after Release tests | Self-contained win-x64 GUI/CLI publish succeeded; solution build zero warnings/errors |
| `git diff --check` | Passed |

The Python skips are optional native/GPU/official-reference cases requiring explicit
environment flags or local fixture paths. This run does not replace historical
fully enabled kernel/reference acceptance evidence. Raw command logs are local,
ignored files under `reports/modeldesk-*-verification.txt` and
`reports/ram-policy-verification.txt`.

## Behavioral evidence

- RAM accepts positive values above 32 GB in Python and C#, survives settings
  persistence and CLI conversion without overflow, and reaches generated Python
  configs. GUI roundtrips one byte through `long.MaxValue` exactly. Live Windows Job
  readback confirms 64,000,000,000 bytes without allocating that amount of RAM.
- Pinned Flash profile discovery uses an independent model/metadata/report location.
  Native architecture checks reject its unsupported architecture/quantization.
- Real ModelRuntimeService → CoreTaskViewModel → RunViewModel integration completes
  two turns with a fake HTTP boundary and preserves user/assistant/user history.
  Native CPU/hybrid delegation and explicit-config precedence remain covered.
- External response tests cover template kwargs, both reasoning fields, completed
  versus empty/token-limited output, cancellation, HTTP errors and profile report listing.
- MCP transport tests cover initialize/initialized, session/protocol headers, paged
  tools, JSON/SSE, opaque image/structured data, malformed messages, limits, denied
  authentication without token/body logging, cancellation/timeout, and explicit
  session recovery without replaying an uncertain action.
- Agent adapter tests cover live schema forwarding, tool/image observations, unknown
  tool rejection, repeated call IDs, tool errors, one call per completion and 12-round
  termination. Manual MCP CLI commands cover parser validation and exit codes.
- Headless WPF tests render all seven tabs in dark/light at 1280×800 and 1100×720,
  with additional external-chat and scrolled-endpoint settings frames. New controls
  were visually inspected. MCP checkbox enables only for external chat and resets
  when returning to native; external direct-answer/native-plan options are disabled.

## Acceptance limits

No full Flash weights, headers or inference were loaded. No production inference
server, OAuth login/refresh or real Jarvis PC control was exercised. Model and MCP
tests use controlled HTTP/service fixtures. The installed serving runtime must
support glm5_next, compressed-tensors INT4, its chat/tool parsers and screenshots.
The pinned profile does not prove the server loaded that revision.

RAM defaults to 32 GB and remains a configurable native committed-memory budget;
it is not a physical-system RAM guarantee or an external-server quota. External
context and hardware allocation belong to the server. Jarvis requires a valid
device-bound OAuth grant and normal Agent authorization. See
[runtime setup](../OPENAI-RUNTIME.md) and [MCP setup](../JARVIS-MCP.md).
