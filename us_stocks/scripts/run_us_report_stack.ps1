param(
    [string]$ConfigPath = "",
    [string]$RepoRoot = "",
    [string]$PythonExe = "python",
    [string]$LogRoot = "",
    [string]$LastYearOutputDir = "us_stocks\artifacts\deep_learning_large_cap_60_dynamic_seq40_clip_q95_last_year",
    [string]$StabilityOutputDir = "us_stocks\artifacts\stability_large_cap_60_dynamic_seq20_clip_q95",
    [string]$SweepOutputDir = "us_stocks\artifacts\transformer_sweep_v2",
    [string]$OutputsRoot = "outputs",
    [switch]$RunSweep
)

$ErrorActionPreference = "Stop"

$defaultUsStocksRoot = Split-Path -Parent $PSScriptRoot
$defaultRepoRoot = Split-Path -Parent $defaultUsStocksRoot
if (-not $RepoRoot) {
    $RepoRoot = $defaultRepoRoot
}
if (-not $ConfigPath) {
    $ConfigPath = Join-Path $defaultUsStocksRoot "config\soft_price_large_cap_60_dynamic_eligibility.yaml"
}
if (-not $LogRoot) {
    $LogRoot = Join-Path $defaultUsStocksRoot "logs"
}

$resolvedRepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
$resolvedConfigPath = [System.IO.Path]::GetFullPath($ConfigPath)
$resolvedLogRoot = [System.IO.Path]::GetFullPath($LogRoot)
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

New-Item -ItemType Directory -Force -Path $resolvedLogRoot | Out-Null
$logPath = Join-Path $resolvedLogRoot "report_stack_$timestamp.log"
New-Item -ItemType File -Force -Path $logPath | Out-Null

function Invoke-LoggedCommand {
    param(
        [string]$StepName,
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "[$StepName]"
    Write-Host "$PythonExe $($Arguments -join ' ')"
    & $PythonExe @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE."
    }
}

try {
    Set-Location $resolvedRepoRoot
    $env:PYTHONPATH = ".\us_stocks\src"
    if (-not $env:LOKY_MAX_CPU_COUNT) {
        $env:LOKY_MAX_CPU_COUNT = "1"
    }

    Write-Host "Repo root: $resolvedRepoRoot"
    Write-Host "Config: $resolvedConfigPath"
    Write-Host "Log: $logPath"
    Write-Host "Latest-year transformer default: seq40 + clip_q95"
    Write-Host "Repeated-window transformer default: seq20 + clip_q95"

    Invoke-LoggedCommand -StepName "Deep learning last-year report" -Arguments @(
        "-m", "us_invest_ai.deep_learning_report",
        "--config", $resolvedConfigPath,
        "--transformer-sequence-lookback-window", "40",
        "--transformer-target-clip-quantile", "0.95",
        "--output-dir", $LastYearOutputDir
    )

    Invoke-LoggedCommand -StepName "Repeated-window stability report" -Arguments @(
        "-m", "us_invest_ai.stability_report",
        "--config", $resolvedConfigPath,
        "--transformer-sequence-lookback-window", "20",
        "--transformer-target-clip-quantile", "0.95",
        "--output-dir", $StabilityOutputDir
    )

    Invoke-LoggedCommand -StepName "Signal hardening analysis" -Arguments @(
        "-m", "us_invest_ai.signal_hardening_report",
        "--last-year-summary", (Join-Path $LastYearOutputDir "deep_learning_summary_last_year.csv"),
        "--stability-summary", (Join-Path $StabilityOutputDir "stability_window_summary.csv"),
        "--leaderboard-output", (Join-Path $OutputsRoot "experiments\signal_hardening\leaderboard.csv"),
        "--analysis-output", (Join-Path $OutputsRoot "reports\performance_degradation\analysis.json"),
        "--guidance-output", (Join-Path $OutputsRoot "reports\signal_hardening\signal_hardening_guidance.md")
    )

    if ($RunSweep) {
        Invoke-LoggedCommand -StepName "Focused transformer sweep" -Arguments @(
            "-m", "us_invest_ai.transformer_sweep",
            "--config", $resolvedConfigPath,
            "--transformer-model-dims", "4",
            "--transformer-training-lookback-days", "252",
            "--sequence-lookback-windows", "10,20,40",
            "--target-clip-quantiles", "none,0.9,0.95",
            "--output-dir", $SweepOutputDir
        )
    }

    Write-Host ""
    Write-Host "Report stack complete."
    Write-Host "Latest-year output: $LastYearOutputDir"
    Write-Host "Stability output: $StabilityOutputDir"
    Write-Host "Signal hardening outputs: $(Join-Path $OutputsRoot 'reports')"
    if ($RunSweep) {
        Write-Host "Sweep output: $SweepOutputDir"
    }
    Write-Host "Log: $logPath"
}
finally {}
