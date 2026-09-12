# GLM Architecture Mapper v0.8.0

The architecture mapper consumes the metadata tensor catalogue only. It never loads tensor payloads.

Command:

```powershell
.\glm.bat architecture-check
```

It classifies tensor names into embedding, attention, MoE expert/router and FP8 scale groups, then produces:

`reports/architecture-latest.json`

The result is structural verification only. Payload values, inference correctness and kernel compatibility remain unverified.
