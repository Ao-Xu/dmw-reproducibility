param(
    [Parameter(Mandatory=$true)]
    [string]$RemoteUrl
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

if (-not (Test-Path ".git")) {
    git init
}

git add .
git commit -m "Initial reproducibility release" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[dmw] No new commit created; continuing with existing history."
}

$existing = git remote
if ($existing -notcontains "origin") {
    git remote add origin $RemoteUrl
} else {
    git remote set-url origin $RemoteUrl
}

git branch -M main
git push -u origin main

