# GLM Flash through an OpenAI-compatible server

ModelDesk 1.2.0 adds the `flash-w4a16` profile for
[`dealignai/GLM-5.3-Flash-CYBERSECURITY-W4A16`](https://huggingface.co/dealignai/GLM-5.3-Flash-CYBERSECURITY-W4A16),
pinned at `4c2ee763ff75f3ba50652ee0f024b0b630cd54a0`.
The existing Python/native decoder does not support its `glm5_next` architecture
or compressed-tensors INT4 format. C# sends structured chat to a separately configured
inference server. OpenAI-compatible describes the HTTP protocol; a local server
requires no OpenAI account or cloud connection.

## Prepare the server

Use a runtime build that supports this model's architecture, quantization, chat
template and vision inputs. The [model card](https://huggingface.co/dealignai/GLM-5.3-Flash-CYBERSECURITY-W4A16)
describes vLLM deployment with the `glm45` reasoning parser and `glm47` tool-call parser.
Its older examples use a different repository alias; use the requested ID and revision
above. ModelDesk does not install/start the server or download weights during chat.

For a compatible vLLM installation and suitable hardware, these are the relevant arguments:

```text
vllm serve dealignai/GLM-5.3-Flash-CYBERSECURITY-W4A16
  --revision 4c2ee763ff75f3ba50652ee0f024b0b630cd54a0
  --served-model-name dealignai/GLM-5.3-Flash-CYBERSECURITY-W4A16
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice
  --max-model-len 8192
```

This lists arguments, not a Windows installation recipe. GPU count, memory allocation
and any offload strategy belong to the chosen runtime. Metadata lists 120 shards
totaling about 194.66 GB; W4A16 does not imply this checkpoint fits a 32 GB PC. No
serving deployment or full-model inference was verified in this patch. See the
[metadata baseline and architecture limits](models/flash-cybersecurity-w4a16/README.md).

## Configure ModelDesk

In Settings, enter the server base URL including `/v1`, optional served model name,
and the environment variable name for its API key if required. The default URL is
`http://127.0.0.1:8000/v1`; a blank model override uses the selected profile's full
Hugging Face ID. Provision credentials in the launching process environment, not
CLI arguments, settings files or chat. Restart after changing that environment.

```powershell
.\modeldesk-cli.bat settings set profile flash-w4a16
.\modeldesk-cli.bat settings set openai-url 'http://127.0.0.1:8000/v1'
.\modeldesk-cli.bat settings set openai-model 'dealignai/GLM-5.3-Flash-CYBERSECURITY-W4A16'
.\modeldesk-cli.bat chat --profile flash-w4a16 --prompt 'Xin chào' --context 8192 --max-tokens 2000
```

Selecting Flash in the GUI chooses `openai`, context 8192, output budget 2000 and
reasoning `low`. Other profiles can use an external server with `--backend openai`.
Native FP8/NVFP4 remain available through `cpu` and `hybrid`.

Requests use `chat_template_kwargs.reasoning_effort` and `clear_thinking`, without
`enable_thinking` (unsupported by the pinned template). Direct-answer prefix injection
belongs to the native decoder and is disabled externally. Thinking still consumes
output tokens with `low`. Both `reasoning` and older `reasoning_content` response fields
are accepted. See [vLLM tool calling](https://docs.vllm.ai/en/latest/features/tool_calling/).

External requests use complete JSON responses (`stream: false`); the answer appears
after the server finishes. Native chat retains token streaming. Only nonempty
`finish_reason: stop` replies count as complete history turns. Token exhaustion,
tool limits, errors, cancellation and timeouts remain incomplete.

## Resources and reports

RAM defaults to **32,000,000,000 bytes** with no former 32 GB maximum. GUI accepts
positive decimal GB, CLI `ram-mib` accepts positive MiB, and Python accepts positive
platform-representable byte budgets. Native planning and Windows Job limits use the
selected value. Changing it does not preallocate RAM. CPU/GPU policies are unchanged.

The separate server manages its RAM, CPU, GPU and context window. Native resource
settings do not constrain it. External `--context` only checks output budget <= that
value; it does not tokenize/truncate history or configure the server context window.
The server must reject requests exceeding its actual capacity.

Receipts and logs appear in the GUI Reports tab under
`reports/<profile>/openai/<run-id>/` (legacy FP8 uses the `reports/` root).
They identify the selected profile pin, but do not verify the external server's loaded
checkpoint revision or establish numerical parity.

[Jarvis MCP setup](JARVIS-MCP.md) enables PC tools with `chat --mcp` or the GUI checkbox.
The adapter permits one PC tool call per completion, at most 12 tool rounds, then asks
for a final answer with tools disabled. Tool results/screenshots return to the model.
Agent authorization remains server-enforced. Calls are not automatically retried;
cancellation cannot undo completed PC actions.
