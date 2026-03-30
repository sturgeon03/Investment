param(
    [string]$RepoRoot = "",
    [string]$PythonExe = "",
    [string]$OvernightConfigPath = "",
    [string]$OllamaDailyConfigPath = "",
    [int]$CutoffHour = 10,
    [int]$CutoffMinute = 0
)

$ErrorActionPreference = "Stop"

function Get-IsoTimestamp {
    return (Get-Date).ToString("o")
}

function Write-StatusFile {
    param(
        [string]$Path,
        [object]$Payload
    )

    $json = $Payload | ConvertTo-Json -Depth 8 -Compress
    Set-Content -Encoding UTF8 -Path $Path -Value $json
}

function Append-Ledger {
    param(
        [string]$Path,
        [object]$Payload
    )

    $json = $Payload | ConvertTo-Json -Depth 8 -Compress
    Add-Content -Encoding UTF8 -Path $Path -Value $json
}

function Get-LastTaskFinishedAt {
    param(
        [string]$LedgerPath,
        [string]$TaskName,
        [datetime]$Fallback
    )

    if (-not (Test-Path $LedgerPath)) {
        return $Fallback
    }

    $pattern = ('\"task_name\":\"' + [regex]::Escape($TaskName) + '\"')
    $lastMatch = Get-Content -Path $LedgerPath | Where-Object { $_ -match $pattern } | Select-Object -Last 1
    if (-not $lastMatch) {
        return $Fallback
    }

    try {
        $entry = $lastMatch | ConvertFrom-Json
        if ($entry.finished_at) {
            return [datetime]::Parse([string]$entry.finished_at)
        }
    }
    catch {
    }

    return $Fallback
}

function Test-OllamaHealthy {
    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" -Method Get -TimeoutSec 5
        return $true
    }
    catch {
        return $false
    }
}

function Ensure-OllamaServer {
    param(
        [string]$RepoRoot
    )

    if (Test-OllamaHealthy) {
        return $true
    }

    $ollamaExe = "C:\Users\sym89\AppData\Local\Programs\Ollama\ollama.exe"
    if (-not (Test-Path $ollamaExe)) {
        return $false
    }

    $logRoot = Join-Path $RepoRoot "us_stocks\logs\ollama"
    New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
    $stdoutPath = Join-Path $logRoot "night_ollama_serve.out.log"
    $stderrPath = Join-Path $logRoot "night_ollama_serve.err.log"

    Start-Process `
        -FilePath $ollamaExe `
        -ArgumentList "serve" `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath | Out-Null

    for ($i = 0; $i -lt 18; $i++) {
        Start-Sleep -Seconds 5
        if (Test-OllamaHealthy) {
            return $true
        }
    }

    return $false
}

function Invoke-LoggedTask {
    param(
        [string]$TaskName,
        [string]$TaskKind,
        [string]$RepoRoot,
        [string]$StateRoot,
        [string]$Executable,
        [string[]]$Arguments
    )

    $latestStatusPath = Join-Path $StateRoot "latest_status.json"
    $ledgerPath = Join-Path $StateRoot "ledger\task_ledger.jsonl"
    $taskLogRoot = Join-Path $StateRoot "logs"
    New-Item -ItemType Directory -Force -Path $taskLogRoot | Out-Null
    $taskLogPath = Join-Path $taskLogRoot ("{0}_{1}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"), $TaskName)

    $start = Get-IsoTimestamp
    $record = [ordered]@{
        job_name = "night_batch_until_10"
        task_name = $TaskName
        task_kind = $TaskKind
        started_at = $start
        finished_at = $null
        success = $false
        exit_code = $null
        log_path = $taskLogPath
        command = "$Executable $($Arguments -join ' ')"
        error = $null
    }

    Write-StatusFile -Path $latestStatusPath -Payload ([ordered]@{
        job_name = "night_batch_until_10"
        active_task = $TaskName
        active_task_kind = $TaskKind
        started_at = $start
        finished_at = $null
        success = $null
        exit_code = $null
        log_path = $taskLogPath
        repo_root = $RepoRoot
        state_root = $StateRoot
    })

    try {
        & $Executable @Arguments 2>&1 | Tee-Object -FilePath $taskLogPath
        $record.exit_code = $LASTEXITCODE
        if ($LASTEXITCODE -ne 0) {
            throw "$TaskName failed with exit code $LASTEXITCODE."
        }
        $record.success = $true
    }
    catch {
        $record.exit_code = if ($record.exit_code -eq $null) { 1 } else { $record.exit_code }
        $record.error = $_.Exception.Message
    }
    finally {
        $record.finished_at = Get-IsoTimestamp
        Append-Ledger -Path $ledgerPath -Payload $record
        Write-StatusFile -Path $latestStatusPath -Payload $record
    }

    return $record
}

$defaultUsStocksRoot = Split-Path -Parent $PSScriptRoot
$defaultRepoRoot = Split-Path -Parent $defaultUsStocksRoot
if (-not $RepoRoot) {
    $RepoRoot = $defaultRepoRoot
}
if (-not $PythonExe) {
    $venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        $PythonExe = $venvPython
    }
    else {
        $PythonExe = "python"
    }
}
if (-not $OvernightConfigPath) {
    $OvernightConfigPath = Join-Path $defaultUsStocksRoot "config\soft_price_large_cap_60_dynamic_eligibility.yaml"
}
if (-not $OllamaDailyConfigPath) {
    $OllamaDailyConfigPath = Join-Path $defaultUsStocksRoot "config\with_llm_ollama_local_night.yaml"
}

$stateRoot = Join-Path $defaultUsStocksRoot "automation\night_shift"
$lockDir = Join-Path $stateRoot "night_until_10.lock"
$latestStatusPath = Join-Path $stateRoot "latest_status.json"
$heartbeatPath = Join-Path $stateRoot "heartbeat.json"
$ledgerPath = Join-Path $stateRoot "ledger\task_ledger.jsonl"
New-Item -ItemType Directory -Force -Path $stateRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $stateRoot "ledger") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $stateRoot "logs") | Out-Null

if (Test-Path $lockDir) {
    Remove-Item -Recurse -Force $lockDir
}
New-Item -ItemType Directory -Force -Path $lockDir | Out-Null

$cutoff = Get-Date -Hour $CutoffHour -Minute $CutoffMinute -Second 0
$reportMinGapMinutes = 150
$paperMinGapMinutes = 70
$nowForFallback = Get-Date
$lastReportFinished = Get-LastTaskFinishedAt `
    -LedgerPath $ledgerPath `
    -TaskName "overnight_prefer_cache" `
    -Fallback $nowForFallback.AddMinutes(-$reportMinGapMinutes)
$lastPaperFinished = Get-LastTaskFinishedAt `
    -LedgerPath $ledgerPath `
    -TaskName "ollama_local_preview" `
    -Fallback $nowForFallback.AddMinutes(-$paperMinGapMinutes)

try {
    while ((Get-Date) -lt $cutoff) {
        $now = Get-Date
        $minutesLeft = ($cutoff - $now).TotalMinutes

        Write-StatusFile -Path $heartbeatPath -Payload ([ordered]@{
            job_name = "night_batch_until_10"
            updated_at = Get-IsoTimestamp
            cutoff = $cutoff.ToString("o")
            minutes_left = [math]::Round($minutesLeft, 2)
            last_report_finished = $lastReportFinished.ToString("o")
            last_paper_finished = $lastPaperFinished.ToString("o")
        })

        if ($minutesLeft -ge 75 -and (($now - $lastReportFinished).TotalMinutes -ge $reportMinGapMinutes)) {
            $record = Invoke-LoggedTask `
                -TaskName "overnight_prefer_cache" `
                -TaskKind "research" `
                -RepoRoot $RepoRoot `
                -StateRoot $stateRoot `
                -Executable "powershell.exe" `
                -Arguments @(
                    "-NoProfile",
                    "-ExecutionPolicy", "Bypass",
                    "-File", (Join-Path $PSScriptRoot "run_overnight_quant_prefer_cache.ps1"),
                    "-RepoRoot", $RepoRoot,
                    "-PythonExe", $PythonExe,
                    "-ConfigPath", $OvernightConfigPath
                )
            $lastReportFinished = [datetime]::Parse($record.finished_at)
            continue
        }

        if ($minutesLeft -ge 15 -and (($now - $lastPaperFinished).TotalMinutes -ge $paperMinGapMinutes)) {
            $ollamaReady = Ensure-OllamaServer -RepoRoot $RepoRoot
            if ($ollamaReady) {
                $record = Invoke-LoggedTask `
                    -TaskName "ollama_local_preview" `
                    -TaskKind "paper_preview" `
                    -RepoRoot $RepoRoot `
                    -StateRoot $stateRoot `
                    -Executable "powershell.exe" `
                    -Arguments @(
                        "-NoProfile",
                        "-ExecutionPolicy", "Bypass",
                        "-File", (Join-Path $PSScriptRoot "run_us_daily.ps1"),
                        "-ConfigPath", $OllamaDailyConfigPath,
                        "-RepoRoot", $RepoRoot,
                        "-PythonExe", $PythonExe,
                        "-LogRoot", (Join-Path $defaultUsStocksRoot "logs\ollama_night"),
                        "-RunLabel", "night_ollama_preview"
                    )
                $lastPaperFinished = [datetime]::Parse($record.finished_at)
                continue
            }

            Append-Ledger -Path (Join-Path $stateRoot "ledger\task_ledger.jsonl") -Payload ([ordered]@{
                job_name = "night_batch_until_10"
                task_name = "ollama_health_check"
                task_kind = "paper_preview"
                started_at = Get-IsoTimestamp
                finished_at = Get-IsoTimestamp
                success = $false
                exit_code = 1
                log_path = $null
                command = "ensure ollama server"
                error = "Ollama server was not healthy and could not be started."
            })
            Start-Sleep -Seconds 300
            continue
        }

        Start-Sleep -Seconds 300
    }

    Write-StatusFile -Path $latestStatusPath -Payload ([ordered]@{
        job_name = "night_batch_until_10"
        active_task = $null
        active_task_kind = $null
        started_at = $null
        finished_at = Get-IsoTimestamp
        success = $true
        exit_code = 0
        log_path = $null
        repo_root = $RepoRoot
        state_root = $stateRoot
        note = "Cutoff reached."
    })
}
finally {
    if (Test-Path $lockDir) {
        Remove-Item -Recurse -Force $lockDir
    }
}
