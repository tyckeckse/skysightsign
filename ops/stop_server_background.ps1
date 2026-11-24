$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -Path $repoRoot

$pidFile = Join-Path $repoRoot "data\app_server.pid"
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$appPort = 8000

function Get-AppServerProcessCimFromPid {
  param([int]$PidValue)
  $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $PidValue" -ErrorAction SilentlyContinue
  if ($null -eq $proc) {
    return $null
  }

  $exePath = "$($proc.ExecutablePath)"
  $cmdLine = "$($proc.CommandLine)"
  $isPython = $exePath -ieq $python
  if (-not $isPython) {
    # The venv python.exe is a launcher that runs the base interpreter as a child process.
    $parent = Get-CimInstance Win32_Process -Filter "ProcessId = $($proc.ParentProcessId)" -ErrorAction SilentlyContinue
    $isPython = $null -ne $parent -and "$($parent.ExecutablePath)" -ieq $python
  }
  $isAppServer = $cmdLine -match "scripts[\\/]+app_server\.py(\s|$)"
  if ($isPython -and $isAppServer) {
    return $proc
  }
  return $null
}

function Get-AppServerListenerCim {
  $listener = Get-NetTCPConnection -LocalAddress "127.0.0.1" -LocalPort $appPort -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
  if ($null -eq $listener) {
    return $null
  }
  return Get-AppServerProcessCimFromPid -PidValue ([int]$listener.OwningProcess)
}

$processesToStop = @{}

if (Test-Path $pidFile) {
  $pidText = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
  if ($pidText -match '^\d+$') {
    $appProc = Get-AppServerProcessCimFromPid -PidValue ([int]$pidText)
    if ($null -ne $appProc) {
      $processesToStop[[string]$appProc.ProcessId] = $appProc
    }
  }
}

$listenerProc = Get-AppServerListenerCim
if ($null -ne $listenerProc) {
  $processesToStop[[string]$listenerProc.ProcessId] = $listenerProc
}

if ($processesToStop.Count -eq 0) {
  if (Test-Path $pidFile) {
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
  }
  Write-Host "Ingen app_server-process hittades."
  exit 0
}

foreach ($proc in $processesToStop.Values) {
  Stop-Process -Id ([int]$proc.ProcessId) -Force -ErrorAction SilentlyContinue
  Write-Host "App server stoppad (PID $($proc.ProcessId))."
}

Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
