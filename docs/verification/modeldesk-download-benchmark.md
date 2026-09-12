# ModelDesk local HTTP download verification

Verified 2026-09-12 at 13:20 UTC, .NET 10 on Windows. The benchmark uses the production downloader and real TCP/HTTP sockets bound only to `127.0.0.1`. No remote weights or credentials were used.

The deterministic test file is 67,108,864 bytes (64 MiB). The server generates 64 KiB blocks by absolute offset, adds 100 ms before each response header, and limits **each connection** to 8 MiB/s. Measurements include final checksum validation.

| Path | Wall time | Throughput | Peak data connections | Payload bytes sent |
|---|---:|---:|---:|---:|
| Sequential | 8.373 s | 7.644 MiB/s | 1 | 67,108,864 |
| Four ranges | 2.338 s | 27.370 MiB/s | 4 | 67,108,865 |

The parallel path includes one extra byte for its Range capability probe. Both files matched SHA-256 `de99c892313522a171e642d6922c5974186a37bebaf03bbf8f048b8fe6b7a403`.

Cancellation after 9 MiB left an identified resumable checkpoint. Restart preserved 9,437,184 bytes and requested exactly 57,671,680 remaining payload bytes, plus one probe byte. The preserved and requested payload sizes sum to the complete file size; the resumed file passed SHA-256 verification. Resume took 2.049 s. The benchmark removed its temporary payloads after completion.

The observed 3.58× speedup applies to this imposed per-connection limit. **It is not an internet or Hugging Face throughput measurement or guarantee.** Host load, disk performance, server policy and network conditions can change results.

Reproduce using [the benchmark instructions](../../studio/benchmarks/ModelDesk.DownloadBench/README.md). Raw local evidence is generated at `reports/studio/download-benchmark-64mib.json`.
