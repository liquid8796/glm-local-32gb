# Local download benchmark

This exercises the production downloader through real HTTP sockets bound only to `127.0.0.1`. The server generates deterministic bytes in 64 KiB blocks, supports exact Range requests and strong ETags, and imposes configurable latency and a per-connection bandwidth limit. No remote model files or credentials are used.

```powershell
dotnet run --project studio/benchmarks/ModelDesk.DownloadBench -c Release -- --size-mib 64 --latency-ms 100 --connection-mib 8 --output reports/studio/download-benchmark-64mib.json
```

To avoid rebuilding assemblies used by a running GUI or test host, build into a separate artifact directory and execute that copy:

```powershell
dotnet build studio/benchmarks/ModelDesk.DownloadBench -c Release --artifacts-path reports/studio/benchbuild
dotnet reports/studio/benchbuild/bin/ModelDesk.DownloadBench/release/modeldesk-download-bench.dll --size-mib 64 --latency-ms 100 --connection-mib 8 --output reports/studio/download-benchmark-64mib.json
```

Use `--size-mib 16` for a shorter run. The benchmark adjusts the segmentation threshold and segment size so the one-connection and four-connection paths both run at either size. Temporary payloads are removed after execution.

JSON output records wall time, throughput, server bytes written, peak data connections, SHA-256 verification, cancellation, and persisted resume-byte accounting. A successful resume must satisfy `persisted_resumed_bytes + requested_remaining_bytes == expected_file_bytes` after excluding one-byte capability probes.

These are controlled local measurements under an artificial **per-connection** server limit. They do not measure or promise Hugging Face internet download speed. The sequential and segmented cases include final checksum validation; OS caching and machine load can affect timings.
