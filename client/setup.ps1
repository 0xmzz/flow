# Flow thin-client installer. Run in PowerShell on the client PC (no admin needed):
#   irm http://<flow-host>:8756/client/setup.ps1 | iex
# The server templates {{URL}} and {{TOKEN}} before serving this file.
$ErrorActionPreference = "Stop"
$URL = "{{URL}}"
$TOKEN = "{{TOKEN}}"
$DIR = "$env:LOCALAPPDATA\Flow"

Write-Host "Flow thin client -> $URL"
New-Item -ItemType Directory -Force $DIR | Out-Null

# 1) Python
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py -or -not (& python --version 2>$null)) {
    Write-Host "Installing Python 3.12 (winget)..."
    winget install -e --id Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements | Out-Null
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
}

# 2) App files
foreach ($f in @("flow.py", "store.py", "llm.py", "commands.py", "supervisor.py", "flow.ico")) {
    Invoke-WebRequest -Uri "$URL/client/file/$f" -OutFile "$DIR\$f" -UseBasicParsing
}

# 3) Venv + light deps (no CUDA, no model)
if (-not (Test-Path "$DIR\.venv")) { python -m venv "$DIR\.venv" }
& "$DIR\.venv\Scripts\pip" install -q keyboard sounddevice numpy pyperclip pystray pillow requests comtypes pycaw

# 4) Config
@{
    mode = "client"; server_url = $URL; client_name = $env:COMPUTERNAME
    hotkey = "right ctrl"; command_hotkey = "right shift"
    server = @{ enabled = $false; host = "0.0.0.0"; port = 8756; token = $TOKEN }
    history = @{ enabled = $false; dir = "data"; archive_audio = $false }
    streaming = @{ enabled = $false; interval_s = 1.2 }
    cleanup = @{ enabled = $false; provider = "ollama"; model = ""; base_url = ""; api_key = ""; api_key_env = "FLOW_API_KEY"; extra_body = @{}; fallback = @{}; profiles = @(); mode = "matched"; min_words = 5 }
    overlay = $true; mute_output_while_recording = $true
} | ConvertTo-Json -Depth 5 | Set-Content "$DIR\config.json" -Encoding utf8

# 5) Shortcuts + autostart — best-effort; a shortcut failure must NOT stop the
# install (Desktop is often redirected to OneDrive, etc.).
$ws = New-Object -ComObject WScript.Shell
# GetFolderPath resolves the REAL locations (handles OneDrive redirection).
$desktop = [Environment]::GetFolderPath("Desktop")
$programs = [Environment]::GetFolderPath("Programs")
$startup = [Environment]::GetFolderPath("Startup")
foreach ($d in @($desktop, $programs)) {
    try {
        if ($d -and (Test-Path $d)) {
            $lnk = $ws.CreateShortcut((Join-Path $d "Flow.lnk"))
            $lnk.TargetPath = "$DIR\.venv\Scripts\pythonw.exe"
            $lnk.Arguments = "`"$DIR\supervisor.py`""
            $lnk.WorkingDirectory = $DIR
            $lnk.IconLocation = "$DIR\flow.ico"
            $lnk.Description = "Flow - voice dictation (hold right ctrl)"
            $lnk.Save()
        }
    } catch { Write-Host "  (skipped shortcut in $d)" -ForegroundColor DarkGray }
}
try {
    if ($startup -and (Test-Path $startup)) {
        @"
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = "$DIR"
sh.Run """$DIR\.venv\Scripts\pythonw.exe"" ""$DIR\supervisor.py""", 0, False
"@ | Set-Content (Join-Path $startup "flow.vbs") -Encoding ascii
    }
} catch { Write-Host "  (skipped autostart entry)" -ForegroundColor DarkGray }

# 6) Launch now (the part that actually matters)
Start-Process "$DIR\.venv\Scripts\pythonw.exe" -ArgumentList "`"$DIR\supervisor.py`"" -WorkingDirectory $DIR
Write-Host ""
Write-Host "Done. Hold RIGHT CTRL and speak - text appears where your cursor is." -ForegroundColor Green
Write-Host "(Look for the mic icon in the system tray. Starts automatically on login.)"
