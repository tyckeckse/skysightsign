Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$tasks = @(
  'SkysightForecastCacheHourly',
  'SkysightForecastCacheAtStartup',
  'SkysightAppServerAtLogon',
  'SkysightFeedbackRetrainNightly'
)

foreach ($task in $tasks) {
  schtasks /Delete /TN $task /F | Out-Null
}

Write-Host 'Removed tasks (if they existed).'
