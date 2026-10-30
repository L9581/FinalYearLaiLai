$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found at $python. Run .\setup_new_computer.ps1 first."
}
& $python -m streamlit run (Join-Path $PSScriptRoot 'robot_agent_app.py') --server.address 127.0.0.1 --server.port 8503 --server.headless true
