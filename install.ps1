# Flow standalone installer: everything runs on this PC's NVIDIA GPU.
# Right-click this file -> "Run with PowerShell" (no admin needed), or:
#   powershell -ExecutionPolicy Bypass -File install.ps1            # dictation only
#   powershell -ExecutionPolicy Bypass -File install.ps1 -Cleanup   # + local LLM tidy-up (Ollama)
param([switch]$Cleanup)
$ErrorActionPreference = "Stop"
$SRC = $PSScriptRoot
$DIR = "$env:LOCALAPPDATA\Flow"
$PY = "$DIR\.venv\Scripts\python.exe"

Write-Host "Installing Flow to $DIR" -ForegroundColor Cyan
New-Item -ItemType Directory -Force $DIR | Out-Null

# 1) NVIDIA driver check (Blackwell cards need a recent driver)
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Host "nvidia-smi not found. Install the latest NVIDIA driver first, then re-run." -ForegroundColor Yellow
    exit 1
}
& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# 2) Python 3.12 (the GPU wheels are pinned against it)
$hasPy312 = $false
try { & py -3.12 --version *> $null; $hasPy312 = ($LASTEXITCODE -eq 0) } catch {}
if (-not $hasPy312) {
    Write-Host "Installing Python 3.12 (winget)..."
    winget install -e --id Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements | Out-Null
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
}

# 3) App files
$APP_FILES = @("flow.py", "agents.py", "commands.py", "export.py", "hermes.py", "hud.py",
    "live.py", "llm.py", "server.py", "speech.py", "store.py", "streaming.py",
    "supervisor.py", "winctl.py", "workflows.py", "flow.ico", "README.md")
foreach ($f in $APP_FILES) { Copy-Item (Join-Path $SRC $f) -Destination $DIR -Force }
# kokoro-onnx is installed separately below (--no-deps)
Get-Content (Join-Path $SRC "requirements.txt") | Where-Object { $_ -notmatch '^kokoro-onnx' } |
    Set-Content "$DIR\requirements.txt" -Encoding ascii

# 4) Venv + deps. kokoro-onnx goes in with --no-deps: it pins plain
# onnxruntime, which would overwrite the GPU build and drop every model to CPU.
if (-not (Test-Path $PY)) { & py -3.12 -m venv "$DIR\.venv" }
& $PY -m pip install -q --upgrade pip
Write-Host "Installing packages (a few GB of CUDA libraries, takes a while)..."
& $PY -m pip install -q -r "$DIR\requirements.txt"
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
& $PY -m pip install -q --no-deps kokoro-onnx==0.5.0
& $PY -m pip install -q phonemizer-fork espeakng-loader

# 5) Optional: Ollama for app-aware cleanup. qwen3:4b, not 8b: an 8 GB card
# can't hold Whisper large-v3 + Parakeet + an 8b model at once.
if ($Cleanup) {
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        Write-Host "Installing Ollama (winget)..."
        winget install -e --id Ollama.Ollama --silent --accept-package-agreements --accept-source-agreements | Out-Null
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
        Start-Sleep 5
    }
    & ollama pull qwen3:4b
}

# 6) Machine check + config (picks the live mic, enables cleanup if Ollama is up)
Push-Location $DIR
& $PY flow.py --setup
Pop-Location
if ($Cleanup -and (Test-Path "$DIR\config.json")) {
    & $PY -c "import json, pathlib; p = pathlib.Path(r'$DIR\config.json'); c = json.loads(p.read_text(encoding='utf-8-sig')); c['cleanup']['model'] = 'qwen3:4b'; p.write_text(json.dumps(c, indent=2), encoding='utf-8')"
}

# 7) Download Whisper large-v3 now (~3 GB) and prove the GPU works, so the
# first launch isn't a silent multi-minute wait.
Write-Host "Downloading the speech model and testing the GPU..."
Push-Location $DIR
& $PY -c "import flow, numpy as np; from faster_whisper import WhisperModel; m = WhisperModel('large-v3', device='cuda', compute_type='float16'); m.transcribe(np.zeros(16000, dtype=np.float32)); print('GPU OK')"
$gpuOk = ($LASTEXITCODE -eq 0)
Pop-Location
if (-not $gpuOk) { Write-Host "GPU test failed - Flow will fall back to CPU (slow). Update the NVIDIA driver and re-run." -ForegroundColor Yellow }

# 8) Shortcuts + autostart (best-effort; Desktop may be redirected to OneDrive)
$ws = New-Object -ComObject WScript.Shell
foreach ($d in @([Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs"))) {
    try {
        if ($d -and (Test-Path $d)) {
            $lnk = $ws.CreateShortcut((Join-Path $d "Flow.lnk"))
            $lnk.TargetPath = "$DIR\.venv\Scripts\pythonw.exe"
            $lnk.Arguments = "`"$DIR\supervisor.py`""
            $lnk.WorkingDirectory = $DIR
            $lnk.IconLocation = "$DIR\flow.ico"
            $lnk.Description = "Flow - voice dictation (hold F8)"
            $lnk.Save()
        }
    } catch { Write-Host "  (skipped shortcut in $d)" -ForegroundColor DarkGray }
}
try {
    $startup = [Environment]::GetFolderPath("Startup")
    @"
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = "$DIR"
sh.Run """$DIR\.venv\Scripts\pythonw.exe"" ""$DIR\supervisor.py""", 0, False
"@ | Set-Content (Join-Path $startup "flow.vbs") -Encoding ascii
} catch { Write-Host "  (skipped autostart entry)" -ForegroundColor DarkGray }

# 9) Launch
Start-Process "$DIR\.venv\Scripts\pythonw.exe" -ArgumentList "`"$DIR\supervisor.py`"" -WorkingDirectory $DIR
Write-Host ""
Write-Host "Done. Hold F8 and speak - text appears where your cursor is." -ForegroundColor Green
Write-Host "Mic icon in the system tray = running. Starts automatically on login."
Write-Host "Settings: $DIR\config.json   Logs: $DIR\flow.log"
