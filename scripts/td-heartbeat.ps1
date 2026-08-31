<#
.SYNOPSIS
    TouchDesigner heartbeat probe -- pings the td-http-api /health bridge on a
    fixed interval and logs the result to a daily CSV. The last successful ping
    is the precise freeze-onset timestamp for the "left it, came back frozen"
    investigation (see TD_FREEZE_INVESTIGATION.md).

.DESCRIPTION
    /health returns {cookRate, webServerTotalCooks, ...}. This probe records,
    every tick:
      - whether the bridge answered, and how fast (latency)
      - cookRate (TD fps) and webServerTotalCooks (a monotonic cook counter)
      - the delta in the cook counter since the previous tick
      - the wall-clock GAP since the previous tick (a big gap = the probe
        process itself was suspended => the machine was in Modern Standby /
        hibernate; this brackets the sleep window)

    Freeze signatures it catches:
      - BRIDGE_DOWN : /health did not answer  => TD main thread frozen (or TD
                      not running / server off). This is the primary silent-
                      freeze detector -- more reliable than process.Responding.
      - COOK_STALL  : /health answered but the cook counter did not advance
                      => TD's cook loop is stalled while the server thread lives.
      - WOKE_GAP    : a resume after a long self-gap (machine was asleep).

    On an OK->frozen transition it prints and logs the LAST-GOOD timestamp and
    the exact correlator command to run:
        .\sleep-wake-correlator.ps1 -FreezeTime "<last good time>"

.PARAMETER IntervalSeconds
    Seconds between pings. Default 5.

.PARAMETER Port
    td-http-api port. Default 3031.

.PARAMETER TimeoutSeconds
    Per-request timeout. Default 4. Anything slower counts as a failed ping.

.EXAMPLE
    .\td-heartbeat.ps1
    .\td-heartbeat.ps1 -IntervalSeconds 10
#>

param(
    [int]$IntervalSeconds = 5,
    [int]$Port = 3031,
    [int]$TimeoutSeconds = 4
)

$ErrorActionPreference = 'SilentlyContinue'
$baseUrl = "http://127.0.0.1:$Port/health"   # 127.0.0.1, never localhost (DNS/IPv6 delay)
$logDir  = Join-Path (Split-Path $PSScriptRoot -Parent) 'logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$alertLog = Join-Path $logDir 'td-heartbeat.log'

function Write-Alert {
    param([string]$Message)
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    Add-Content -Path $alertLog -Value $line
}

$header = 'timestamp,gap_s,ok,latency_ms,http_status,cook_rate,total_cooks,cook_delta,state'

function Get-CsvPath {
    Join-Path $logDir ("td-heartbeat_{0}.csv" -f (Get-Date -Format 'yyyy-MM-dd'))
}

Write-Host "  TD HEARTBEAT " -BackgroundColor DarkBlue -ForegroundColor White -NoNewline
Write-Host "  probing $baseUrl every ${IntervalSeconds}s  |  CSV: logs\td-heartbeat_<date>.csv  |  Ctrl+C to stop" -ForegroundColor DarkGray
Write-Host ""
Write-Alert "Heartbeat started. interval=${IntervalSeconds}s port=$Port timeout=${TimeoutSeconds}s"

$prevTime   = $null
$prevCooks  = $null
$prevState  = 'START'
$lastGood   = $null

while ($true) {
    $now = Get-Date
    $ts  = $now.ToString('yyyy-MM-dd HH:mm:ss')

    # Wall-clock gap since previous tick (detects the probe being frozen by standby)
    $gap = if ($prevTime) { [math]::Round(($now - $prevTime).TotalSeconds, 1) } else { 0 }
    $wokeFromGap = ($prevTime -and $gap -gt ($IntervalSeconds * 3))

    # Ping /health
    $ok = $false; $latency = $null; $httpStatus = ''; $cookRate = ''; $totalCooks = $null
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $resp = Invoke-WebRequest -Uri $baseUrl -UseBasicParsing -TimeoutSec $TimeoutSeconds
        $sw.Stop()
        $latency = [math]::Round($sw.Elapsed.TotalMilliseconds)
        $httpStatus = $resp.StatusCode
        $data = $resp.Content | ConvertFrom-Json
        $cookRate = $data.cookRate
        $totalCooks = [int64]$data.webServerTotalCooks
        $ok = $true
    } catch {
        $sw.Stop()
        $latency = [math]::Round($sw.Elapsed.TotalMilliseconds)
        $httpStatus = 'ERR'
    }

    # Cook delta
    $cookDelta = ''
    if ($ok -and $null -ne $prevCooks -and $null -ne $totalCooks) {
        $cookDelta = $totalCooks - $prevCooks
    }

    # Determine state
    if (-not $ok) {
        $state = 'BRIDGE_DOWN'
    } elseif ($cookDelta -ne '' -and $cookDelta -le 0) {
        $state = 'COOK_STALL'
    } elseif ($wokeFromGap) {
        $state = 'WOKE_GAP'
    } else {
        $state = 'OK'
    }

    # CSV append (write header if new/rotated file)
    $csvPath = Get-CsvPath
    if (-not (Test-Path $csvPath)) { Add-Content -Path $csvPath -Value $header }
    $row = "{0},{1},{2},{3},{4},{5},{6},{7},{8}" -f `
        $ts, $gap, $ok, $latency, $httpStatus, $cookRate, $totalCooks, $cookDelta, $state
    Add-Content -Path $csvPath -Value $row

    # Console line
    $color = switch ($state) {
        'OK'          { 'Green' }
        'WOKE_GAP'    { 'Cyan' }
        'COOK_STALL'  { 'Yellow' }
        'BRIDGE_DOWN' { 'Red' }
        default       { 'Gray' }
    }
    $detail = if ($ok) { "fps=$cookRate cooks=$totalCooks (+$cookDelta) ${latency}ms" } else { "no response (${latency}ms timeout)" }
    if ($wokeFromGap) { $detail += "  [gap ${gap}s <- machine was asleep]" }
    Write-Host ("  {0}  {1,-11}  {2}" -f $now.ToString('HH:mm:ss'), $state, $detail) -ForegroundColor $color

    # Transition detection -> pinpoint freeze onset
    $isFrozen = ($state -eq 'BRIDGE_DOWN' -or $state -eq 'COOK_STALL')
    $wasFrozen = ($prevState -eq 'BRIDGE_DOWN' -or $prevState -eq 'COOK_STALL')

    if ($isFrozen -and -not $wasFrozen) {
        $onset = if ($lastGood) { $lastGood.ToString('yyyy-MM-dd HH:mm:ss') } else { 'unknown' }
        Write-Host ""
        Write-Host "  *** FREEZE DETECTED ($state) ***" -ForegroundColor Red
        Write-Host "  Last good ping: $onset" -ForegroundColor Red
        Write-Host "  Correlate it:   .\scripts\sleep-wake-correlator.ps1 -FreezeTime `"$onset`"" -ForegroundColor Yellow
        Write-Host ""
        Write-Alert "FREEZE DETECTED state=$state lastGood=$onset gap=${gap}s"
    }
    if ($wasFrozen -and -not $isFrozen) {
        Write-Host "  >> bridge recovered" -ForegroundColor Green
        Write-Alert "RECOVERED state=$state"
    }
    if ($wokeFromGap) {
        Write-Alert "WOKE_GAP gap=${gap}s (probe suspended -> machine asleep) state=$state"
    }

    # Advance
    if ($ok -and $state -eq 'OK') { $lastGood = $now }
    $prevTime  = $now
    if ($ok) { $prevCooks = $totalCooks }
    $prevState = $state

    Start-Sleep -Seconds $IntervalSeconds
}
