param(
    [string]$RepoRoot = "",
    [string]$PythonExe = "",
    [string]$ConfigPath = ""
)

$ErrorActionPreference = "Stop"

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
if (-not $ConfigPath) {
    $ConfigPath = Join-Path $defaultUsStocksRoot "config\soft_price_large_cap_60_dynamic_eligibility.yaml"
}

& (Join-Path $PSScriptRoot "run_overnight_quant.ps1") `
    -RepoRoot $RepoRoot `
    -PythonExe $PythonExe `
    -ConfigPath $ConfigPath `
    -PreferCache `
    -SkipPromoteCanonical

exit $LASTEXITCODE
