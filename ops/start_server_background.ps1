$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -Path $repoRoot

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$appScriptRel = "scripts\app_server.py"
$appPort = 8000
$appBaseUrl = "http://127.0.0.1:$appPort"
if (!(Test-Path $python)) {
  throw "Saknar $python. Skapa/aktivera .venv fÃ¶rst."
}

$dataDir = Join-Path $repoRoot "data"
$pidFile = Join-Path $dataDir "app_server.pid"
$outLogFile = Join-Path $dataDir "app_server.out.log"
$errLogFile = Join-Path $dataDir "app_server.err.log"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

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

function Stop-AppServerProcess {
  param([Microsoft.Management.Infrastructure.CimInstance]$Proc)
  if ($null -eq $Proc) {
    return
  }
  Stop-Process -Id ([int]$Proc.ProcessId) -Force -ErrorAction SilentlyContinue
}

function Test-AppServerHealthy {
  try {
    $resp = Invoke-WebRequest -Uri "$appBaseUrl/api/health" -UseBasicParsing -TimeoutSec 3
    return $resp.StatusCode -eq 200
  } catch {
    return $false
  }
}

$processesToStop = @{}

if (Test-Path $pidFile) {
  $existingPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
  if ($existingPid -match '^\d+$') {
    $existingCim = Get-AppServerProcessCimFromPid -PidValue ([int]$existingPid)
    if ($null -ne $existingCim) {
      $processesToStop[[string]$existingCim.ProcessId] = $existingCim
    }
  }
  Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

$listenerCim = Get-AppServerListenerCim
if ($null -ne $listenerCim) {
  $processesToStop[[string]$listenerCim.ProcessId] = $listenerCim
}

foreach ($proc in $processesToStop.Values) {
  Stop-AppServerProcess -Proc $proc
}

Start-Sleep -Milliseconds 500

$portListener = Get-NetTCPConnection -LocalAddress "127.0.0.1" -LocalPort $appPort -State Listen -ErrorAction SilentlyContinue |
  Select-Object -First 1
if ($null -ne $portListener) {
  $blockingPid = [int]$portListener.OwningProcess
  $blockingProc = Get-CimInstance Win32_Process -Filter "ProcessId = $blockingPid" -ErrorAction SilentlyContinue
  $blockingName = if ($null -ne $blockingProc) { "$($blockingProc.Name)" } else { "okÃ¤nd process" }
  throw "Port $appPort anvÃ¤nds redan av PID $blockingPid ($blockingName)."
}

$proc = Start-Process -FilePath $python `
  -ArgumentList @("-u", $appScriptRel) `
  -WorkingDirectory $repoRoot `
  -RedirectStandardOutput $outLogFile `
  -RedirectStandardError $errLogFile `
  -WindowStyle Hidden `
  -PassThru

Set-Content -Path $pidFile -Value $proc.Id -Encoding ASCII

$startedHealthy = $false
for ($i = 0; $i -lt 10; $i++) {
  Start-Sleep -Seconds 1
  $alive = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
  if ($null -eq $alive) {
    break
  }
  # The listening socket may belong to the launcher's child interpreter, not $proc itself.
  $childIds = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $($proc.Id)" -ErrorAction SilentlyContinue | ForEach-Object ProcessId)
  $listener = Get-NetTCPConnection -LocalAddress "127.0.0.1" -LocalPort $appPort -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.OwningProcess -eq $proc.Id -or $childIds -contains $_.OwningProcess } |
    Select-Object -First 1
  if ($null -ne $listener -and (Test-AppServerHealthy)) {
    $startedHealthy = $true
    break
  }
}

if (-not $startedHealthy) {
  $errTail = if (Test-Path $errLogFile) { (Get-Content $errLogFile -Tail 50) -join [Environment]::NewLine } else { "" }
  $outTail = if (Test-Path $outLogFile) { (Get-Content $outLogFile -Tail 50) -join [Environment]::NewLine } else { "" }
  throw "App server start failed or unhealthy. PID $($proc.Id). ERR: $errTail OUT: $outTail"
}

Write-Host "App server started in background (PID $($proc.Id))."
Write-Host "URL: $appBaseUrl/"
