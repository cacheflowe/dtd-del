<#
.SYNOPSIS
    System flight recorder -- appends one row of health metrics per tick to a
    daily CSV so an unattended TouchDesigner freeze is fully reconstructable
    after the fact. The persistent-history companion to td-watchdog.ps1 (which
    only keeps 20 readings in RAM) and to td-heartbeat.ps1 (which watches the
    bridge). See TD_FREEZE_INVESTIGATION.md.

.DESCRIPTION
    Every tick it records GPU temp / P-state / clock / util / power, NVDisplay
    user-session handle count, nonpaged pool, RAM, and TD process state
    (running / responding / thread count / suspended-thread count / handles /
    working set). It also records the wall-clock GAP since the previous tick:
    a gap far larger than the interval means the recorder itself was suspended,
    which brackets a Modern Standby / hibernate window.

    When you return to a frozen TD, open the CSV and read the rows around the
    wake: you get the precise tick where TD flipped responding TRUE->FALSE, the
    suspended-thread count at that moment (OS-suspension vs self-deadlock), and
    the GPU P-state across the sleep boundary.

.PARAMETER IntervalSeconds
    Seconds between samples. Default 15.

.PARAMETER KillAsusHandlesAbove
    If > 0, auto-kill the leaking asus_framework.exe instance when its handle
    count exceeds this threshold (it respawns). 0 = disabled (default) -- just
    log the count. The ArmouryDevice backend leaks handles unbounded at roughly
    ~4.7k/hr; note this does NOT drive nonpaged pool (verified 2026-07-20).

.EXAMPLE
    .\td-flight-recorder.ps1
    .\td-flight-recorder.ps1 -IntervalSeconds 30
    .\td-flight-recorder.ps1 -KillAsusHandlesAbove 200000
#>

param(
    [int]$IntervalSeconds = 15,
    [int]$KillAsusHandlesAbove = 0
)

$ErrorActionPreference = 'SilentlyContinue'
$logDir = Join-Path (Split-Path $PSScriptRoot -Parent) 'logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$alertLog = Join-Path $logDir 'td-flight.log'
$hasNvidiaSmi = $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)

function Write-Alert {
    param([string]$Message)
    Add-Content -Path $alertLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
}

$header = 'timestamp,gap_s,gpu_temp,gpu_pstate,gpu_clock_mhz,gpu_util,gpu_power_w,nvdisplay_handles,asus_fw_handles,pool_mb,ram_used_gb,td_running,td_responding,td_threads,td_suspended,td_handles,td_ws_mb,note'

function Get-CsvPath {
    Join-Path $logDir ("td-flight_{0}.csv" -f (Get-Date -Format 'yyyy-MM-dd'))
}

Write-Host "  TD FLIGHT RECORDER " -BackgroundColor DarkBlue -ForegroundColor White -NoNewline
Write-Host "  sampling every ${IntervalSeconds}s  |  CSV: logs\td-flight_<date>.csv  |  Ctrl+C to stop" -ForegroundColor DarkGray
Write-Host ""
Write-Alert "Flight recorder started. interval=${IntervalSeconds}s nvidia-smi=$hasNvidiaSmi"

$prevTime      = $null
$prevResponding = $null

while ($true) {
    $now = Get-Date
    $ts  = $now.ToString('yyyy-MM-dd HH:mm:ss')
    $gap = if ($prevTime) { [math]::Round(($now - $prevTime).TotalSeconds, 1) } else { 0 }
    $wokeFromGap = ($prevTime -and $gap -gt ($IntervalSeconds * 3))

    # GPU
    $gpuTemp = ''; $gpuPstate = ''; $gpuClock = ''; $gpuUtil = ''; $gpuPower = ''
    if ($hasNvidiaSmi) {
        $smi = nvidia-smi --query-gpu=temperature.gpu,pstate,clocks.gr,utilization.gpu,power.draw --format=csv,noheader,nounits 2>$null
        if ($smi) {
            $p = $smi -split ',\s*'
            $gpuTemp = $p[0].Trim(); $gpuPstate = $p[1].Trim(); $gpuClock = $p[2].Trim()
            $gpuUtil = $p[3].Trim(); $gpuPower = $p[4].Trim()
        }
    }

    # NVDisplay user-session handles
    $nvProcs = Get-CimInstance Win32_Process -Filter "Name='NVDisplay.Container.exe'" -EA SilentlyContinue
    $nvUser = $nvProcs | Where-Object { $_.SessionId -ne 0 } | Select-Object -First 1
    $nvUserHandles = if ($nvUser) { $nvUser.HandleCount } else { '' }

    # asus_framework leaking instance (the single highest-handle instance)
    $asusFw = Get-Process -Name asus_framework -EA SilentlyContinue | Sort-Object HandleCount -Descending | Select-Object -First 1
    $asusFwHandles = if ($asusFw) { $asusFw.HandleCount } else { '' }
    $asusKilled = $false
    if ($asusFw -and $KillAsusHandlesAbove -gt 0 -and $asusFw.HandleCount -gt $KillAsusHandlesAbove) {
        try { Stop-Process -Id $asusFw.Id -Force -EA Stop; $asusKilled = $true } catch { }
    }

    # Nonpaged pool + RAM
    $poolMB = ''
    $poolBytes = (Get-Counter '\Memory\Pool Nonpaged Bytes' -EA SilentlyContinue).CounterSamples[0].CookedValue
    if ($poolBytes) { $poolMB = [math]::Round($poolBytes / 1MB) }
    $os = Get-CimInstance Win32_OperatingSystem
    $ramUsed = if ($os) { [math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1MB, 1) } else { '' }

    # TD process
    $tdRunning = $false; $tdResp = ''; $tdThreads = ''; $tdSusp = ''; $tdHandles = ''; $tdWS = ''
    $tdProc = Get-Process -Name "TouchDesigner" -EA SilentlyContinue | Select-Object -First 1
    if ($tdProc) {
        $tdRunning = $true
        $tdThreads = $tdProc.Threads.Count
        $tdSusp = ($tdProc.Threads | Where-Object { $_.WaitReason -eq 'Suspended' } | Measure-Object).Count
        $tdHandles = $tdProc.HandleCount
        $tdWS = [math]::Round($tdProc.WorkingSet64 / 1MB)
        $tdResp = $tdProc.Responding
    }

    # Note column
    $notes = @()
    if ($wokeFromGap) { $notes += "WOKE_GAP:${gap}s" }
    if ($tdRunning -and $tdResp -eq $false) { $notes += 'TD_FROZEN' }
    if ($asusKilled) { $notes += "ASUS_FW_KILLED:$asusFwHandles" }
    $note = $notes -join ';'

    # CSV
    $csvPath = Get-CsvPath
    if (-not (Test-Path $csvPath)) { Add-Content -Path $csvPath -Value $header }
    $row = "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17}" -f `
        $ts, $gap, $gpuTemp, $gpuPstate, $gpuClock, $gpuUtil, $gpuPower, `
        $nvUserHandles, $asusFwHandles, $poolMB, $ramUsed, $tdRunning, $tdResp, $tdThreads, $tdSusp, $tdHandles, $tdWS, $note
    Add-Content -Path $csvPath -Value $row

    # Console line
    $color = 'Green'
    if (-not $tdRunning) { $color = 'DarkGray' }
    elseif ($tdResp -eq $false) { $color = 'Red' }
    elseif ($wokeFromGap) { $color = 'Cyan' }
    $tdStr = if (-not $tdRunning) { 'TD:off' } elseif ($tdResp -eq $false) { "TD:FROZEN($tdSusp/$tdThreads susp)" } else { "TD:ok" }
    $line = "  {0}  GPU:{1}C {2} {3}MHz  NVh:{4}  asusFw:{5}  pool:{6}MB  {7}" -f `
        $now.ToString('HH:mm:ss'), $gpuTemp, $gpuPstate, $gpuClock, $nvUserHandles, $asusFwHandles, $poolMB, $tdStr
    if ($wokeFromGap) { $line += "  [gap ${gap}s]" }
    if ($asusKilled) { $line += "  [asus_fw killed]" }
    Write-Host $line -ForegroundColor $color

    # Transition alert
    if ($tdRunning -and $tdResp -eq $false -and $prevResponding -eq $true) {
        Write-Host "  *** TD froze this tick ($ts). Suspended $tdSusp/$tdThreads threads, GPU $gpuPstate ***" -ForegroundColor Red
        Write-Alert "TD FROZEN at $ts suspended=$tdSusp/$tdThreads pstate=$gpuPstate gap=${gap}s"
    }
    if ($wokeFromGap) { Write-Alert "WOKE_GAP gap=${gap}s pstate=$gpuPstate td_responding=$tdResp" }
    if ($asusKilled) { Write-Host "  *** killed leaking asus_framework (handles=$asusFwHandles > $KillAsusHandlesAbove) ***" -ForegroundColor Yellow; Write-Alert "ASUS_FW_KILLED handles=$asusFwHandles threshold=$KillAsusHandlesAbove" }

    $prevTime = $now
    if ($tdRunning) { $prevResponding = $tdResp }
    Start-Sleep -Seconds $IntervalSeconds
}
