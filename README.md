# Skysightsign

Minimal local workflow (Windows/PowerShell).

## Start
```powershell
cd <path-to-skysightsign>
. .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python .\scripts\train_model.py
.\ops\start_server_background.ps1
```

Open `http://127.0.0.1:8000/`.

Run in foreground:
```powershell
python .\app.py
```

## Background Tasks
Install or update tasks:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\ops\install_tasks.ps1
```

Remove tasks:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\ops\remove_tasks.ps1
```

Installed task names:
- `SkysightForecastCacheHourly`
- `SkysightForecastCacheAtStartup`
- `SkysightAppServerAtLogon`
- `SkysightFeedbackRetrainNightly`

## Published page
GitHub Actions (`.github/workflows/skysight.yml`) updates https://skysightsign.vercel.app hourly, 04:37-21:37 UTC, and trains at 11:37 UTC.
Models, `mlflow.db` and the forecast log are stored in the private `state` release.
Requires the repository secrets `VERCEL_TOKEN`, `VERCEL_ORG_ID` and `VERCEL_PROJECT_ID`.

## Logs
- `data\train_model.log`
- `data\app_server.log`
- `data\cache_refresh.log`
- `data\app_server.out.log`
- `data\app_server.err.log`

## Credits
Weather icons: Yr/NRK, CC BY 4.0. Forecast data: Open-Meteo. Observations: SMHI.
