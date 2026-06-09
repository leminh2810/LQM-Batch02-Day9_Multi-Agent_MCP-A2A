$ErrorActionPreference = "Stop"

# Start all Legal Multi-Agent System services on Windows PowerShell.
# Registry must be first, then leaf agents, then orchestrators.

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $Root "logs"
$PidFile = Join-Path $LogDir "services.pids"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
if (Test-Path $PidFile) {
    Remove-Item -Path $PidFile
}

function Start-AgentService {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Name,
        [Parameter(Mandatory = $true)]
        [string] $Module,
        [Parameter(Mandatory = $true)]
        [int] $Port
    )

    $stdout = Join-Path $LogDir "$Name.out.log"
    $stderr = Join-Path $LogDir "$Name.err.log"

    Write-Host "Starting $Name on port $Port..."
    $process = Start-Process `
        -FilePath "uv" `
        -ArgumentList @("run", "python", "-m", $Module) `
        -WorkingDirectory $Root `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru

    "$Name,$($process.Id),$Port" | Add-Content -Path $PidFile
    return $process
}

Start-AgentService -Name "registry" -Module "registry" -Port 10000 | Out-Null
Start-Sleep -Seconds 2

Start-AgentService -Name "tax_agent" -Module "tax_agent" -Port 10102 | Out-Null
Start-AgentService -Name "compliance_agent" -Module "compliance_agent" -Port 10103 | Out-Null
Start-Sleep -Seconds 3

Start-AgentService -Name "law_agent" -Module "law_agent" -Port 10101 | Out-Null
Start-Sleep -Seconds 3

Start-AgentService -Name "customer_agent" -Module "customer_agent" -Port 10100 | Out-Null

Write-Host ""
Write-Host "All services started:"
Write-Host "  Registry:         http://localhost:10000"
Write-Host "  Customer Agent:   http://localhost:10100"
Write-Host "  Law Agent:        http://localhost:10101"
Write-Host "  Tax Agent:        http://localhost:10102"
Write-Host "  Compliance Agent: http://localhost:10103"
Write-Host ""
Write-Host "Logs:"
Write-Host "  $LogDir"
Write-Host ""
Write-Host "Run the test client:"
Write-Host "  uv run python test_client.py"
Write-Host ""
Write-Host "Stop services:"
Write-Host "  Get-Content logs\services.pids | ForEach-Object { Stop-Process -Id `$_.Split(',')[1] }"
