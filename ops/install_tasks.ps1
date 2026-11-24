Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'
$cacheScript = Join-Path $repoRoot 'scripts\update_forecast_cache.py'
$serverScript = Join-Path $repoRoot 'ops\start_server_background.ps1'
$retrainScript = Join-Path $repoRoot 'ops\run_retrain_background.ps1'

$hourlyTask = 'SkysightForecastCacheHourly'
$startupTask = 'SkysightForecastCacheAtStartup'
$serverTask = 'SkysightAppServerAtLogon'
$feedbackTask = 'SkysightFeedbackRetrainNightly'

$cacheCmd = "`"$pythonExe`" `"$cacheScript`" --hours 24"
$serverCmd = "powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$serverScript`""
$feedbackCmd = "powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$retrainScript`""

function Invoke-SchtasksOrFail {
  param([string]$CommandLine)
  cmd /c $CommandLine
  if ($LASTEXITCODE -ne 0) {
    throw "schtasks failed ($LASTEXITCODE): $CommandLine"
  }
}

if (!(Test-Path $pythonExe)) {
  throw "Saknar $pythonExe. Skapa/aktivera .venv först."
}
if (!(Test-Path $cacheScript)) {
  throw "Saknar $cacheScript."
}
if (!(Test-Path $retrainScript)) {
  throw "Saknar $retrainScript."
}
if (!(Test-Path $serverScript)) {
  throw "Saknar $serverScript."
}

Invoke-SchtasksOrFail "schtasks /Create /TN `"$hourlyTask`" /SC HOURLY /MO 1 /ST 00:00 /TR `"$cacheCmd`" /RU SYSTEM /RL HIGHEST /F"
Invoke-SchtasksOrFail "schtasks /Create /TN `"$startupTask`" /SC ONSTART /TR `"$cacheCmd`" /RU SYSTEM /RL HIGHEST /F"
Invoke-SchtasksOrFail "schtasks /Create /TN `"$serverTask`" /SC ONLOGON /TR `"$serverCmd`" /RU SYSTEM /RL HIGHEST /F"
Invoke-SchtasksOrFail "schtasks /Create /TN `"$feedbackTask`" /SC DAILY /ST 12:00 /TR `"$feedbackCmd`" /RU SYSTEM /RL HIGHEST /F"

Invoke-SchtasksOrFail "schtasks /Run /TN `"$hourlyTask`""
Invoke-SchtasksOrFail "schtasks /Run /TN `"$serverTask`""

Write-Host "Installed tasks: $hourlyTask, $startupTask, $serverTask, $feedbackTask"
Write-Host 'Cache refresh and app server tasks started.'
