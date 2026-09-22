$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    throw 'Python 3.11 or 3.12 is required. Install it from python.org and enable the Add Python to PATH option.'
}

$pythonVersion = & $pythonCommand.Source --version 2>&1
Write-Host "Using $pythonVersion"

$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    & $pythonCommand.Source -m venv (Join-Path $PSScriptRoot '.venv')
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -e '.[whisper,tts,denoise,dev]'

Write-Host ''
Write-Host 'Base setup is complete.'
Write-Host 'Install Ollama, then run: ollama pull qwen2.5:3b'
Write-Host 'Start the app with: .\start_voice_lab.ps1'
Write-Host 'If PowerShell blocks scripts, run:'
Write-Host '  powershell -ExecutionPolicy Bypass -File .\start_voice_lab.ps1'
