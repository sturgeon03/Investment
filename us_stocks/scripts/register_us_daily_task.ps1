param(
    [string]$TaskName = "USStocksDailyWorkflow",
    [string]$StartTime = "22:30",
    [string]$ConfigPath = "C:\Users\sym89\Desktop\Investment\us_stocks\config\with_llm_deepseek.yaml",
    [string]$RepoRoot = "C:\Users\sym89\Desktop\Investment",
    [string]$PythonExe = "python",
    [string]$LogRoot = "C:\Users\sym89\Desktop\Investment\us_stocks\logs",
    [string]$RunLabel = "scheduled",
    [string]$PaperBrokerBackend = "local",
    [string]$PaperBrokerEnvFile = "",
    [switch]$PaperBrokerLiveReadinessCheck,
    [string]$MaxPaperOrderCount = "",
    [string]$MaxPaperTotalTradeNotional = "",
    [string]$MaxPaperSingleOrderNotional = "",
    [switch]$AllowDuplicatePaperSubmission,
    [switch]$ApplyPaperOrders,
    [switch]$SubmitPaperOrders
)

$ErrorActionPreference = "Stop"

$scriptPath = Join-Path $RepoRoot "us_stocks\scripts\run_us_daily.ps1"
$applySwitch = if ($ApplyPaperOrders) { " -ApplyPaperOrders" } else { "" }
$submitSwitch = if ($SubmitPaperOrders) { " -SubmitPaperOrders" } else { "" }
$backendArgs = if ($SubmitPaperOrders) { " -PaperBrokerBackend `"$PaperBrokerBackend`"" } else { "" }
$envFileArgs = if ($PaperBrokerEnvFile) { " -PaperBrokerEnvFile `"$PaperBrokerEnvFile`"" } else { "" }
$liveReadinessArgs = if ($PaperBrokerLiveReadinessCheck) { " -PaperBrokerLiveReadinessCheck" } else { "" }
$maxOrderArgs = if ($MaxPaperOrderCount) { " -MaxPaperOrderCount `"$MaxPaperOrderCount`"" } else { "" }
$maxTotalArgs = if ($MaxPaperTotalTradeNotional) { " -MaxPaperTotalTradeNotional `"$MaxPaperTotalTradeNotional`"" } else { "" }
$maxSingleArgs = if ($MaxPaperSingleOrderNotional) { " -MaxPaperSingleOrderNotional `"$MaxPaperSingleOrderNotional`"" } else { "" }
$duplicateArgs = if ($AllowDuplicatePaperSubmission) { " -AllowDuplicatePaperSubmission" } else { "" }
$taskCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" -ConfigPath `"$ConfigPath`" -RepoRoot `"$RepoRoot`" -PythonExe `"$PythonExe`" -LogRoot `"$LogRoot`" -RunLabel `"$RunLabel`"$applySwitch$submitSwitch$backendArgs$envFileArgs$liveReadinessArgs$maxOrderArgs$maxTotalArgs$maxSingleArgs$duplicateArgs"

schtasks /Create /SC DAILY /TN $TaskName /TR $taskCommand /ST $StartTime /F
