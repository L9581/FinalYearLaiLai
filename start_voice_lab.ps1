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
$hasNvidia = $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)

if ((Test-Path -LiteralPath $voiceUpgradeConfig) -and
    (Test-Path -LiteralPath $voiceBundle) -and
    $hasNvidia) {
    $voiceSettings = & $python -c "import json, sys, tomllib; c=tomllib.load(open(sys.argv[1], 'rb'))['tts']; print(json.dumps({'reference_audio': c['gpt_sovits']['reference_audio'], 'rvc_enabled': c['rvc']['enabled']}))" $voiceUpgradeConfig | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw 'Could not read GPT-SoVITS settings from config.voice-upgrade.toml.' }
    $referenceAudio = $voiceSettings.reference_audio
    if (-not [IO.Path]::IsPathRooted($referenceAudio)) {
        $referenceAudio = Join-Path $PSScriptRoot $referenceAudio
    }
    if (-not (Test-Path -LiteralPath $referenceAudio)) {
        throw "GPT-SoVITS reference WAV not found: $referenceAudio"
    }
    $env:VOICE_LAB_CONFIG = $voiceUpgradeConfig
    $voiceAppPort = 8502
    if ($voiceSettings.rvc_enabled) {
        & $python (Join-Path $PSScriptRoot 'scripts\start_voice_services.py')
    } else {
        & $python (Join-Path $PSScriptRoot 'scripts\start_voice_services.py') --gpt-sovits
    }
    if ($LASTEXITCODE -ne 0) { throw 'Voice service startup failed. Check outputs/voice_upgrade/ logs.' }
} elseif (Test-Path -LiteralPath $localConfig) {
    $env:VOICE_LAB_CONFIG = $localConfig
    $voiceAppPort = 8501
} else {
    $env:VOICE_LAB_CONFIG = $baseConfig
    $voiceAppPort = 8501
}

& $python -m streamlit run (Join-Path $PSScriptRoot 'app.py') --server.address 127.0.0.1 --server.port $voiceAppPort --server.headless true
