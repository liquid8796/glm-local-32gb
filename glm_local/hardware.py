"""Read-only Windows hardware detection. No clock, power or driver changes."""

import csv
import io
import json
import math
import os
import shutil
import subprocess


def _number(value, multiplier=1):
    try:
        number = float(value.strip()) * multiplier
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def parse_nvidia_csv(raw):
    gpus = []
    for row in csv.reader(io.StringIO(raw)):
        if not row:
            continue
        if len(row) != 7:
            raise ValueError("Unexpected nvidia-smi CSV schema")
        gpus.append({
            "index": int(row[0]), "name": row[1].strip(),
            "memory_total_bytes": _number(row[2], 1024**2),
            "memory_free_bytes": _number(row[3], 1024**2),
            "utilization_percent": _number(row[4]),
            "compute_capability": row[5].strip(), "driver": row[6].strip(),
        })
    return gpus


def read_gpus():
    exe = shutil.which("nvidia-smi")
    if not exe:
        raise RuntimeError("nvidia-smi is unavailable")
    result = subprocess.run([
        exe,
        "--query-gpu=index,name,memory.total,memory.free,utilization.gpu,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, timeout=10, check=True)
    return parse_nvidia_csv(result.stdout)


def detect_hardware():
    result = {"platform": os.name, "logical_processors": os.cpu_count(), "errors": []}
    if os.name != "nt":
        result["errors"].append("This hardware probe currently supports Windows only")
        return result
    script = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$cpu = @(Get-CimInstance Win32_Processor)
$osInfo = Get-CimInstance Win32_OperatingSystem
$disks = @(Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | ForEach-Object {
    @{root=$_.DeviceID+'\'; total_bytes=[long]$_.Size; free_bytes=[long]$_.FreeSpace}
})
@{cpu_name=($cpu.Name -join ', '); physical_cores=($cpu | Measure-Object NumberOfCores -Sum).Sum;
physical_memory_bytes=[long]$osInfo.TotalVisibleMemorySize*1024;
available_memory_bytes=[long]$osInfo.FreePhysicalMemory*1024;
disks=$disks} | ConvertTo-Json -Depth 5 -Compress
"""
    try:
        raw = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                             capture_output=True, timeout=20, check=True)
        result.update(json.loads(raw.stdout.decode("utf-8-sig")))
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        result["errors"].append(f"Windows hardware probe failed: {type(error).__name__}")
    try:
        result["gpus"] = read_gpus()
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as error:
        result["gpus"] = []
        result["errors"].append(f"GPU probe failed: {type(error).__name__}")
    return result
