# Jarvis MCP in ModelDesk

ModelDesk connects to the Jarvis MCP server over **Streamable HTTP**, then routes model tool calls to the OAuth-selected Windows Agent. The Python/native inference core is unchanged. Model-driven tools require the configured OpenAI-compatible inference backend; `mcp tools` and `mcp call` can also inspect or invoke MCP directly.

## Connection and credentials

The MCP endpoint is blank by default. Set the URL of your own deployment, including `/mcp`. The Jarvis source at `D:\Project\tools\Jarvis\jarvis-platform` documents `https://jarvis-mcp.158.180.59.36.sslip.io/mcp` as a deployment example; this patch has **not verified that live deployment**. Jarvis exposes `/mcp` over HTTP, not a desktop-control stdio command. Its stdio configuration concerns downstream integrations such as Unity and Blender.

```powershell
.\modeldesk-cli.bat settings set mcp-url 'https://your-jarvis-host/mcp'
.\modeldesk-cli.bat settings set mcp-token-env MODEL_DESK_JARVIS_ACCESS_TOKEN
.\modeldesk-cli.bat settings set openai-url 'http://127.0.0.1:8000/v1'
.\modeldesk-cli.bat settings set openai-model 'your-served-model-name'
.\modeldesk-cli.bat settings set openai-key-env MODEL_DESK_OPENAI_API_KEY
```

The two `*-env` settings contain **environment variable names**, never credentials. Provision their values securely in the process environment before launching ModelDesk. Do not paste tokens into chat, tracked files, command arguments or logs. The MCP token must be an OAuth access token issued by Jarvis for the exact MCP resource and selected device. An Agent enrollment token, device UUID or Hugging Face token cannot authenticate MCP requests.

ModelDesk does not implement OAuth sign-in, registration, consent or token refresh. Provision the token through an authorized OAuth client outside ModelDesk. Jarvis's source documents code flow with PKCE S256, scopes `mcp:tools offline_access`, and an administrator-approved exact callback URI in `Jarvis:OAuthRedirectUris`. It documents a ten-minute access-token lifetime. Replace expired grants securely and restart ModelDesk when changing the launching environment; another process cannot update an already running application's inherited environment.

In the GUI, open **Settings → Model server và Jarvis MCP**, enter the inference base URL, optional served model name, MCP endpoint and credential environment variable names, then save. On the chat/run screen select the external `openai` backend and enable the Jarvis MCP option. The `flash-w4a16` profile selects the external backend automatically. The inference server controls its model RAM/GPU allocation; ModelDesk's native resource limits do not constrain it.

## Discovery and execution

```powershell
.\modeldesk-cli.bat --json mcp tools
.\modeldesk-cli.bat --json mcp call computer_use --arguments 'C:\path\list-windows.json'
.\modeldesk-cli.bat chat --backend openai --mcp --prompt 'Inspect the open application windows.'
```

For the direct call example, create a UTF-8 JSON object file containing `{"action":"list_windows"}`. Use the public tool name returned by discovery. Direct calls print the raw MCP result; a tool-level `isError: true` returns exit code 2, while connection or input errors return 1. These calls execute against the configured device after normal Agent authorization.

The model adapter exposes the live `computer_use` / `computer_use.computer_use` schema. If direct discovery is unavailable it exposes Jarvis's permanent `jarvis__tool_search` and `jarvis__tool_call` gateways; model calls through the latter are restricted to the canonical PC tool ID `computer_use.computer_use`. Each completion may request one tool call, with at most twelve tool rounds per turn. The client does not automatically retry uncertain HTTP/tool actions. MCP text, structured results and screenshots are retained; screenshots are forwarded as image content to the inference server. Image understanding requires a served model that supports image inputs. For a text-only model, use accessibility observations (`include_text: true`, `include_screenshot: false`) and current element indexes.

Jarvis retains control of OAuth device routing, **Arm/Pause**, tool approvals, application grants, denied-app policy and Windows integrity restrictions. ModelDesk does not arm the Agent or grant application access. Choose a window returned by discovery and call `get_window_state` before input. Every input action consumes its current `observation_id`; coordinate actions, scrolling and dragging also require the matching `screenshot_id`. Observe again after an action or uncertain result. `return_state` can request a fresh observation after input.

The client supports JSON and SSE replies, MCP transport session/version headers and paginated tools. It limits requests to 256 KiB and responses/catalogs to 8 MiB, uses per-request timeouts/cancellation, and rejects non-loopback cleartext HTTP. Canceling a request does not roll back actions already dispatched.

## Evidence and source

The regression suite uses fake HTTP transports and fake model/MCP services. It verifies protocol exchanges, tool/image result preservation, error handling, bounds and cancellation. It does not prove production OAuth, enrolled-device availability, model quality or live PC control. No production Jarvis MCP tool was invoked during this implementation.

Jarvis source contracts: `jarvis-mcp-server/src/Jarvis.McpServer/Program.cs` (`MapMcp`, HTTP transport), `docs/OAUTH-MCP.md` (OAuth), `jarvis-agent/src/Jarvis.Agent.Windows/CodexComputerUseTool.cs` (live action schema), `docs/COMPUTER-USE-OBSERVATION-V2.md` (observations), and `jarvis-mcp-server/src/Jarvis.McpServer/Transport/DynamicAgentToolGateway.cs` (fallback tools).

The screenshot message uses a text tool result followed by a user message with `image_url` data URLs, following the [vLLM multimodal input format](https://docs.vllm.ai/en/latest/features/multimodal_inputs/). Acceptance still depends on the served model, chat template and server version.
