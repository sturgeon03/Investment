param(
    [string]$ConfigPath = "",
    [string]$RepoRoot = "",
    [string]$PythonExe = "",
    [string]$StateRoot = "",
    [int]$LockStaleAfterMinutes = 180,
    [switch]$PreferCache,
    [switch]$WaitUntilUsClose,
    [switch]$SkipPromoteCanonical
)

$ErrorActionPreference = "Stop"

function Get-IsoTimestamp {
    return (Get-Date).ToString("o")
}

function Write-LockHeartbeat {
    param(
        [string]$LockDir,
        [string]$Stage
    )

    $payload = [ordered]@{
        updated_at = Get-IsoTimestamp
        stage = $Stage
        pid = $PID
    }
    $payload | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 -Path (Join-Path $LockDir "lock.json")
}

function Wait-UntilUsCloseKst {
    $now = Get-Date
    $target = Get-Date -Hour 5 -Minute 15 -Second 0
    if ($now -ge $target) {
        return
    }

    $seconds = [Math]::Ceiling(($target - $now).TotalSeconds)
    if ($seconds -gt 0) {
        Write-Host "Waiting until 05:15 local time before refreshing US daily data."
        Start-Sleep -Seconds $seconds
    }
}

$defaultUsStocksRoot = Split-Path -Parent $PSScriptRoot
$defaultRepoRoot = Split-Path -Parent $defaultUsStocksRoot
if (-not $RepoRoot) {
    $RepoRoot = $defaultRepoRoot
}

$resolvedRepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
if (-not $ConfigPath) {
    $ConfigPath = Join-Path $defaultUsStocksRoot "config\soft_price_large_cap_60_dynamic_eligibility.yaml"
}
$resolvedConfigPath = [System.IO.Path]::GetFullPath($ConfigPath)

if (-not $PythonExe) {
    $venvPython = Join-Path $resolvedRepoRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        $PythonExe = $venvPython
    }
    else {
        $PythonExe = "python"
    }
}

if (-not $StateRoot) {
    $StateRoot = Join-Path $defaultUsStocksRoot "automation"
}
$resolvedStateRoot = [System.IO.Path]::GetFullPath($StateRoot)
$locksRoot = Join-Path $resolvedStateRoot "locks"
$ledgerRoot = Join-Path $resolvedStateRoot "ledger"
$runsRoot = Join-Path $resolvedStateRoot "runs"
$lockDir = Join-Path $locksRoot "overnight_quant.lock"
$ledgerPath = Join-Path $ledgerRoot "run_ledger.jsonl"
$latestStatusPath = Join-Path $resolvedStateRoot "latest_status.json"

New-Item -ItemType Directory -Force -Path $locksRoot, $ledgerRoot, $runsRoot | Out-Null

if (Test-Path $lockDir) {
    $lockAgeMinutes = ((Get-Date) - (Get-Item $lockDir).LastWriteTime).TotalMinutes
    if ($lockAgeMinutes -lt $LockStaleAfterMinutes) {
        Write-Host "Another overnight run is still active. Exiting without starting a duplicate run."
        exit 0
    }

    Remove-Item -Recurse -Force $lockDir
}

New-Item -ItemType Directory -Force -Path $lockDir | Out-Null
Write-LockHeartbeat -LockDir $lockDir -Stage "starting"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runRoot = Join-Path $runsRoot $timestamp
$logRoot = Join-Path $runRoot "logs"
$artifactRoot = Join-Path $runRoot "artifacts"
$lastYearOutputDir = Join-Path $artifactRoot "deep_learning_last_year"
$stabilityOutputDir = Join-Path $artifactRoot "stability"
$repoHealthPath = Join-Path $runRoot "repo_health.txt"
$summaryPath = Join-Path $runRoot "run_summary.json"
$refreshLogPath = Join-Path $logRoot "refresh_market_data.log"
$canonicalLastYearOutputDir = Join-Path $defaultUsStocksRoot "artifacts\deep_learning_large_cap_60_dynamic_seq40_clip_q95_last_year"
$canonicalStabilityOutputDir = Join-Path $defaultUsStocksRoot "artifacts\stability_large_cap_60_dynamic_seq20_clip_q95"

New-Item -ItemType Directory -Force -Path $runRoot, $logRoot, $artifactRoot | Out-Null

$startedAt = Get-IsoTimestamp
$success = $false
$errorMessage = $null
$refreshStatus = "not_started"
$reportStatus = "not_started"
$repoHealthStatus = [ordered]@{
    status = "not_started"
    exit_code = $null
    log_path = $repoHealthPath
    last_output = $null
}
$latestMarketDate = $null
$marketDataSource = $null
$refreshManifestPath = $null
$promotionStatus = if ($SkipPromoteCanonical) { "skipped" } else { "not_started" }

try {
    Set-Location $resolvedRepoRoot
    $env:PYTHONPATH = ".\us_stocks\src"
    & $PythonExe -m us_invest_ai.repo_health --repo-root $resolvedRepoRoot --output-path $repoHealthPath | Out-Null
    $repoHealthStatus.exit_code = $LASTEXITCODE
    $repoHealthStatus.status = if ($LASTEXITCODE -eq 0) { "succeeded" } else { "warning" }
    if (Test-Path $repoHealthPath) {
        $repoHealthStatus.last_output = (Get-Content -Path $repoHealthPath | Select-Object -Last 1)
    }

    if ($WaitUntilUsClose) {
        Write-LockHeartbeat -LockDir $lockDir -Stage "waiting_for_us_close"
        Wait-UntilUsCloseKst
    }

    Write-LockHeartbeat -LockDir $lockDir -Stage "refresh_market_data"
    $refreshStatus = "running"
    $refreshArgs = @(
        "-m", "us_invest_ai.refresh_market_data",
        "--config", $resolvedConfigPath
    )
    if ($PreferCache) {
        $refreshArgs += "--prefer-cache"
    }
    & $PythonExe @refreshArgs 2>&1 | Tee-Object -FilePath $refreshLogPath
    if ($LASTEXITCODE -ne 0) {
        throw "refresh_market_data failed with exit code $LASTEXITCODE."
    }
    $refreshStatus = "succeeded"

    $configDataDir = & $PythonExe -c "from us_invest_ai.config import load_config; print(load_config(r'$resolvedConfigPath').output.data_dir)"
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to resolve config output.data_dir."
    }
    $resolvedDataDir = [System.IO.Path]::GetFullPath(($configDataDir | Select-Object -Last 1).Trim())
    $refreshManifestPath = Join-Path $resolvedDataDir "raw\refresh_run_manifest.json"
    $marketDataManifestPath = Join-Path $resolvedDataDir "raw\market_data_manifest.json"
    if (Test-Path $marketDataManifestPath) {
        $marketDataManifest = Get-Content -Raw -Path $marketDataManifestPath | ConvertFrom-Json
        $latestMarketDate = $marketDataManifest.prices_summary.end_date
        $marketDataSource = $marketDataManifest.source
    }

    Write-LockHeartbeat -LockDir $lockDir -Stage "run_report_stack"
    $reportStatus = "running"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $defaultUsStocksRoot "scripts\run_us_report_stack.ps1") `
        -RepoRoot $resolvedRepoRoot `
        -PythonExe $PythonExe `
        -ConfigPath $resolvedConfigPath `
        -LogRoot $logRoot `
        -LastYearOutputDir $lastYearOutputDir `
        -StabilityOutputDir $stabilityOutputDir
    if ($LASTEXITCODE -ne 0) {
        throw "run_us_report_stack.ps1 failed with exit code $LASTEXITCODE."
    }
    $reportStatus = "succeeded"

    if (-not $SkipPromoteCanonical) {
        Write-LockHeartbeat -LockDir $lockDir -Stage "promote_canonical_artifacts"
        $promotionStatus = "running"
        $promoteArgs = @(
            "-m", "us_invest_ai.promote_report_stack_outputs",
            "--last-year-src", $lastYearOutputDir,
            "--last-year-dest", $canonicalLastYearOutputDir,
            "--stability-src", $stabilityOutputDir,
            "--stability-dest", $canonicalStabilityOutputDir
        )
        & $PythonExe @promoteArgs 2>&1 | Tee-Object -FilePath (Join-Path $logRoot "promote_canonical.log")
        if ($LASTEXITCODE -ne 0) {
            throw "promote_report_stack_outputs failed with exit code $LASTEXITCODE."
        }
        $promotionStatus = "succeeded"
    }

    $success = $true
}
catch {
    $errorMessage = $_.Exception.Message
    if ([string]::IsNullOrWhiteSpace($errorMessage)) {
        $errorMessage = ($_ | Out-String).Trim()
    }
    if ($refreshStatus -eq "running") {
        $refreshStatus = "failed"
    }
    if ($reportStatus -eq "running") {
        $reportStatus = "failed"
    }
    if ($promotionStatus -eq "running") {
        $promotionStatus = "failed"
    }
}
finally {
    $finishedAt = Get-IsoTimestamp
    $summary = [ordered]@{
        job_name = "overnight_quant_orchestrator"
        started_at = $startedAt
        finished_at = $finishedAt
        success = $success
        error = $errorMessage
        repo_root = $resolvedRepoRoot
        config_path = $resolvedConfigPath
        python_exe = $PythonExe
        run_root = $runRoot
        logs = [ordered]@{
            refresh_market_data = $refreshLogPath
            report_stack_root = $logRoot
            repo_health = $repoHealthPath
        }
        output_dirs = [ordered]@{
            last_year_report = $lastYearOutputDir
            stability_report = $stabilityOutputDir
            canonical_last_year_report = $canonicalLastYearOutputDir
            canonical_stability_report = $canonicalStabilityOutputDir
        }
        refresh_manifest_path = $refreshManifestPath
        latest_market_date = $latestMarketDate
        market_data_source = $marketDataSource
        refresh_status = $refreshStatus
        report_status = $reportStatus
        repo_health_status = $repoHealthStatus
        promotion_status = $promotionStatus
        next_recommended_action = if ($success) {
            "Review the latest overnight report artifacts and continue the next highest-priority validated task."
        }
        else {
            "Inspect the overnight logs, fix the blocking error, and rerun the overnight orchestrator."
        }
    }

    $summary | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 -Path $summaryPath
    $summary | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 -Path $latestStatusPath
    ($summary | ConvertTo-Json -Compress -Depth 8) | Add-Content -Encoding UTF8 -Path $ledgerPath

    if (Test-Path $lockDir) {
        Remove-Item -Recurse -Force $lockDir
    }

    if (-not $success) {
        Write-Error $errorMessage
        exit 1
    }
}
