$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found at $python. Run .\setup_new_computer.ps1 first."
}

$voiceUpgradeConfig = Join-Path $PSScriptRoot 'config.voice-upgrade.toml'
$localConfig = Join-Path $PSScriptRoot 'config.local.toml'
$baseConfig = Join-Path $PSScriptRoot 'config.toml'
$voiceBundle = Join-Path $PSScriptRoot 'vitsModel\GPT-SoVITS-v2pro-20250604-nvidia50\runtime\python.exe'
$rvcModel = Join-Path $PSScriptRoot 'rvc\RVC_Nahida\nahida.pth'
$referenceAudio = Join-Path $PSScriptRoot 'data\tts_reference\kokoro_female.wav'
$hasNvidia = $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)

if ((Test-Path -LiteralPath $voiceUpgradeConfig) -and
    (Test-Path -LiteralPath $voiceBundle) -and
    (Test-Path -LiteralPath $rvcModel) -and
    (Test-Path -LiteralPath $referenceAudio) -and
    $hasNvidia) {
    $env:VOICE_LAB_CONFIG = $voiceUpgradeConfig
    $voiceAppPort = 8502
    & $python (Join-Path $PSScriptRoot 'scripts\start_voice_services.py')
    if ($LASTEXITCODE -ne 0) { throw 'Voice service startup failed. Check outputs/voice_upgrade/ logs.' }
} elseif (Test-Path -LiteralPath $localConfig) {
    $env:VOICE_LAB_CONFIG = $localConfig
    $voiceAppPort = 8501
} else {
    $env:VOICE_LAB_CONFIG = $baseConfig
    $voiceAppPort = 8501
}

& $python -m streamlit run (Join-Path $PSScriptRoot 'app.py') --server.address 127.0.0.1 --server.port $voiceAppPort --server.headless true
