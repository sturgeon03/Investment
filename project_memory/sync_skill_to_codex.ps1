param(
    [string]$RepoRoot = "C:\Users\sym89\Desktop\Investment",
    [string]$SkillName = "ai-quant-research-skill",
    [string]$CodexSkillsRoot = "C:\Users\sym89\.codex\skills"
)

$ErrorActionPreference = "Stop"

$source = Join-Path $RepoRoot "project_memory\$SkillName"
$destination = Join-Path $CodexSkillsRoot $SkillName

if (-not (Test-Path $source)) {
    throw "Source skill folder not found: $source"
}

if (Test-Path $destination) {
    Remove-Item -Recurse -Force $destination
}

Copy-Item -Recurse -Force $source $destination
Write-Host "Synced skill to $destination"
