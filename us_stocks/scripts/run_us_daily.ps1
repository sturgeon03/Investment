param(
    [string]$ConfigPath = "",
    [string]$RepoRoot = "",
    [string]$PythonExe = "",
    [string]$LogRoot = "",
    [string]$RunLabel = "",
    [string]$Provider = "",
    [string]$PositionsCsv = "",
    [switch]$ApplyPaperOrders
)

$ErrorActionPreference = "Stop"

$defaultUsStocksRoot = Split-Path -Parent $PSScriptRoot
$defaultRepoRoot = Split-Path -Parent $defaultUsStocksRoot
if (-not $RepoRoot) {
    $RepoRoot = $defaultRepoRoot
}
if (-not $ConfigPath) {
    $ConfigPath = Join-Path $defaultUsStocksRoot "config\with_llm_swing.yaml"
}
if (-not $LogRoot) {
    $LogRoot = Join-Path $defaultUsStocksRoot "logs"
}

$resolvedRepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
$resolvedConfigPath = [System.IO.Path]::GetFullPath($ConfigPath)
$resolvedLogRoot = [System.IO.Path]::GetFullPath($LogRoot)
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

if (-not $PythonExe) {
    $venvPython = Join-Path $resolvedRepoRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        $PythonExe = $venvPython
    }
    else {
        $PythonExe = "python"
    }
}

New-Item -ItemType Directory -Force -Path $resolvedLogRoot | Out-Null
$logPath = Join-Path $resolvedLogRoot "daily_$timestamp.log"

Set-Location $resolvedRepoRoot
$env:PYTHONPATH = ".\us_stocks\src"

if (-not $PositionsCsv) {
    $resolvedPositionsPath = & $PythonExe -c "from us_invest_ai.config import load_config; print(load_config(r'$resolvedConfigPath').workflow.positions_path)"
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to resolve workflow.positions_path from config."
    }
    $PositionsCsv = ($resolvedPositionsPath | Select-Object -Last 1).Trim()
}
$resolvedPositionsCsv = [System.IO.Path]::GetFullPath($PositionsCsv)

$arguments = @(
    "-m", "us_invest_ai.daily_workflow",
    "--config", $resolvedConfigPath,
    "--positions-csv", $resolvedPositionsCsv
)

if ($RunLabel) {
    $arguments += @("--run-label", $RunLabel)
}
if ($Provider) {
    $arguments += @("--provider", $Provider)
}
if ($ApplyPaperOrders) {
    $arguments += "--apply-paper-orders"
}

Write-Host "Repo root: $resolvedRepoRoot"
Write-Host "Config: $resolvedConfigPath"
Write-Host "Log: $logPath"
Write-Host "Positions CSV: $resolvedPositionsCsv"

& $PythonExe @arguments 2>&1 | Tee-Object -FilePath $logPath
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$statusArguments = @(
    "-m", "us_invest_ai.paper_runtime_status",
    "--positions-path", $resolvedPositionsCsv
)

Write-Host ""
Write-Host "[Paper runtime status]"
& $PythonExe @statusArguments 2>&1 | Tee-Object -FilePath $logPath -Append
exit $LASTEXITCODE
