param(
    [string]$ConfigPath = "C:\Users\sym89\Desktop\Investment\us_stocks\config\with_llm_deepseek.yaml",
    [string]$RepoRoot = "C:\Users\sym89\Desktop\Investment",
    [string]$PythonExe = "python",
    [string]$LogRoot = "C:\Users\sym89\Desktop\Investment\us_stocks\logs",
    [string]$RunLabel = "",
    [switch]$ApplyPaperOrders
)

$ErrorActionPreference = "Stop"

$resolvedRepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
$resolvedConfigPath = [System.IO.Path]::GetFullPath($ConfigPath)
$resolvedLogRoot = [System.IO.Path]::GetFullPath($LogRoot)
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

New-Item -ItemType Directory -Force -Path $resolvedLogRoot | Out-Null
$logPath = Join-Path $resolvedLogRoot "daily_$timestamp.log"

Set-Location $resolvedRepoRoot
$env:PYTHONPATH = ".\us_stocks\src"

$arguments = @(
    "-m", "us_invest_ai.daily_workflow",
    "--config", $resolvedConfigPath
)

if ($RunLabel) {
    $arguments += @("--run-label", $RunLabel)
}
if ($ApplyPaperOrders) {
    $arguments += "--apply-paper-orders"
}

Write-Host "Repo root: $resolvedRepoRoot"
Write-Host "Config: $resolvedConfigPath"
Write-Host "Log: $logPath"

& $PythonExe @arguments 2>&1 | Tee-Object -FilePath $logPath
exit $LASTEXITCODE
