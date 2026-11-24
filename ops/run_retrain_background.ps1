$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -Path $repoRoot

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$trainScript = Join-Path $repoRoot "scripts\train_model.py"
$dataDir = Join-Path $repoRoot "data"
$logFile = Join-Path $dataDir "train_model.log"

function Get-UtcIsoTimestamp {
  return (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
}

function Escape-LogValue {
  param([string]$Value)
  if ($null -eq $Value) {
    return "null"
  }
  if ($Value -eq "") {
    return '""'
  }
  if ($Value.IndexOfAny([char[]]' ="') -ge 0) {
    $escaped = $Value.Replace('"', "'")
    return '"' + $escaped + '"'
  }
  return $Value
}

function Write-WrapperLog {
  param(
    [string]$Level,
    [string]$Event,
    [hashtable]$Fields
  )

  $parts = @(
    "ts=$(Get-UtcIsoTimestamp)",
    "level=$Level",
    "svc=scheduler",
    "event=$Event"
  )

  if ($null -ne $Fields) {
    foreach ($key in $Fields.Keys) {
      $value = "$($Fields[$key])"
      $parts += "$key=$(Escape-LogValue $value)"
    }
  }

  Add-Content -Path $logFile -Value ($parts -join " ")
}

if (!(Test-Path $python)) {
  throw "Saknar $python. Skapa/aktivera .venv först."
}
if (!(Test-Path $trainScript)) {
  throw "Saknar $trainScript."
}

New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

Write-WrapperLog -Level "INFO" -Event "retrain_wrapper_start" -Fields @{
  script = "run_retrain_background.ps1"
}

try {
  & $python -u $trainScript `
    --history-days 1400 `
    --rounds 100 `
    --early-stopping-rounds 30 `
    --log-file $logFile `
    --log-level INFO `
    --log-eval-every 10

  $exitCode = $LASTEXITCODE
  if ($exitCode -eq 0) {
    Write-WrapperLog -Level "INFO" -Event "retrain_wrapper_end" -Fields @{
      status = "ok"
      exit_code = $exitCode
    }
  } else {
    Write-WrapperLog -Level "ERROR" -Event "retrain_wrapper_end" -Fields @{
      status = "failed"
      exit_code = $exitCode
    }
    exit $exitCode
  }
} catch {
  Write-WrapperLog -Level "ERROR" -Event "retrain_wrapper_failed" -Fields @{
    status = "failed"
    error = $_.Exception.Message
  }
  throw
}
