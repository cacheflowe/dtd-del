<#
.SYNOPSIS
    Sleep/wake <-> TouchDesigner-freeze correlator. Builds a timeline of
    system sleep, resume, and Modern Standby events and lines them up against
    TD freeze / hang / crash indicators to test the hypothesis that TD
    deadlocks across a sleep/wake cycle.

    Companion to power-audit.ps1 (which tells you HOW the box sleeps). This
    tells you WHAT happened around each freeze.

.DESCRIPTION
    For the "left the computer, came back to a frozen TD" bug documented in
    _assets/TD_CRASHES.md. Pulls, over the scan window:
      - Kernel-Power 42   (entering sleep)      / 107 (resume)
      - Kernel-Power 506  (enter Modern Standby)/ 507 (exit Modern Standby)
      - Power-Troubleshooter 1 (sleep time, wake time, WAKE SOURCE)
      - Kernel-Power 41   (dirty/unexpected reboot)
    ...merges them into one chronological timeline, then reports TD's current
    state and correlates each TD hang/crash (and the current freeze, if any)
    with the nearest preceding wake event.

.PARAMETER HoursBack
    How far back to scan, in hours. Default 48.

.PARAMETER FreezeTime
    Optional explicit freeze time ("yyyy-MM-dd HH:mm" or any parseable form)
    to correlate against, if you know when TD stopped responding. If omitted,
    the script uses TD's current non-responding state and historical WER hang
    events.

.EXAMPLE
    .\sleep-wake-correlator.ps1
    .\sleep-wake-correlator.ps1 -HoursBack 12
    .\sleep-wake-correlator.ps1 -FreezeTime "2026-07-20 03:14"
#>

param(
    [double]$HoursBack = 48,
    [string]$FreezeTime
)

# -- Helpers ---------------------------------------------------------------

function Write-Color {
    param([string]$Text, [string]$Color = 'White')
    Write-Host $Text -ForegroundColor $Color -NoNewline
}

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "  $Title" -ForegroundColor White
}

function Write-Finding {
    param([string]$Label, [string]$Value, [string]$Color = 'Gray')
    Write-Host "    $Label " -ForegroundColor DarkGray -NoNewline
    Write-Host $Value -ForegroundColor $Color
}

function Format-Delta {
    param([TimeSpan]$Span)
    if ($Span.TotalSeconds -lt 0) { return "-" + (Format-Delta $Span.Negate()) }
    if ($Span.TotalMinutes -lt 1) { return "{0}s" -f [math]::Round($Span.TotalSeconds) }
    if ($Span.TotalHours -lt 1)   { return "{0}m {1}s" -f [math]::Floor($Span.TotalMinutes), $Span.Seconds }
    return "{0}h {1}m" -f [math]::Floor($Span.TotalHours), $Span.Minutes
}

$now = Get-Date
$scanStart = $now.AddHours(-$HoursBack)

Clear-Host
Write-Host "  SLEEP/WAKE CORRELATOR " -BackgroundColor DarkBlue -ForegroundColor White -NoNewline
Write-Host "  $($now.ToString('yyyy-MM-dd HH:mm:ss'))  |  scan: last ${HoursBack}h" -ForegroundColor DarkGray

# -- Gather power events ---------------------------------------------------

$events = [System.Collections.ArrayList]@()

function Add-Ev {
    param($Time, [string]$Kind, [string]$Detail, [string]$Color = 'Gray')
    [void]$events.Add([PSCustomObject]@{ Time = $Time; Kind = $Kind; Detail = $Detail; Color = $Color })
}

# Kernel-Power 42 / 107 / 506 / 507 / 41
$kp = Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-Kernel-Power'; Id=42,107,506,507,41; StartTime=$scanStart} -MaxEvents 500 -EA SilentlyContinue
foreach ($e in $kp) {
    switch ($e.Id) {
        42  { Add-Ev $e.TimeCreated 'SLEEP'       'Entering sleep' 'Yellow' }
        107 { Add-Ev $e.TimeCreated 'RESUME'      'Resumed from sleep' 'Cyan' }
        506 { Add-Ev $e.TimeCreated 'MS-ENTER'    'Entering Modern Standby' 'Yellow' }
        507 { Add-Ev $e.TimeCreated 'MS-EXIT'     'Exiting Modern Standby' 'Cyan' }
        41  { Add-Ev $e.TimeCreated 'DIRTY-REBOOT' 'Kernel-Power 41 (unexpected shutdown/reboot)' 'Red' }
    }
}

# Power-Troubleshooter 1 -- richest source: sleep time, wake time, wake source
$pt = Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-Power-Troubleshooter'; Id=1; StartTime=$scanStart} -MaxEvents 200 -EA SilentlyContinue
foreach ($e in $pt) {
    $m = $e.Message -replace "`r`n|`n", ' | '
    $src = if ($e.Message -match 'Wake Source:\s*(.+)') { $Matches[1].Trim() } else { 'unknown' }
    $sleepT = if ($e.Message -match 'Sleep Time:\s*(.+)') { $Matches[1].Trim() } else { $null }
    $detail = "Wake source: $src"
    if ($sleepT) { $detail += "  (slept: $sleepT)" }
    Add-Ev $e.TimeCreated 'WAKE-DETAIL' $detail 'Cyan'
}

$sorted = $events | Sort-Object Time

# -- Current TD state ------------------------------------------------------

Write-Section "TOUCHDESIGNER STATE"

$tdProc = Get-Process -Name "TouchDesigner" -EA SilentlyContinue | Select-Object -First 1
$tdFrozenNow = $false
$tdFreezeAnchor = $null
if (-not $tdProc) {
    Write-Finding "Process:" "NOT RUNNING" 'DarkGray'
} else {
    $threads = $tdProc.Threads.Count
    $suspended = ($tdProc.Threads | Where-Object { $_.WaitReason -eq 'Suspended' } | Measure-Object).Count
    $ageH = [math]::Round(($now - $tdProc.StartTime).TotalHours, 1)
    Write-Finding "PID:" "$($tdProc.Id)   started $($tdProc.StartTime.ToString('yyyy-MM-dd HH:mm:ss'))  (age ${ageH}h)" 'Gray'
    if (-not $tdProc.Responding) {
        $tdFrozenNow = $true
        Write-Finding "Responding:" "FALSE -- TD IS FROZEN RIGHT NOW" 'Red'
        Write-Finding "Threads:" "$threads total, $suspended suspended" $(if ($suspended -gt ($threads - 3)) { 'Red' } else { 'Yellow' })
        if ($suspended -gt ($threads - 3)) {
            Write-Host "    -> almost all threads Suspended = process was suspended (Modern" -ForegroundColor DarkGray
            Write-Host "       Standby / Connected Standby) and did not fully resume." -ForegroundColor DarkGray
        } else {
            Write-Host "    -> threads not mass-suspended = more like a self-deadlock (e.g." -ForegroundColor DarkGray
            Write-Host "       render thread waiting on a lost GPU fence) than OS suspension." -ForegroundColor DarkGray
        }
    } else {
        Write-Finding "Responding:" "TRUE (OK)" 'Green'
        Write-Finding "Threads:" "$threads total, $suspended suspended" 'Gray'
    }
}

# Rough "last known TD activity" from today's app log tail
$logDir = Join-Path (Split-Path $PSScriptRoot -Parent) 'logs'
$todayLog = Join-Path $logDir ("td-app_{0}.txt" -f $now.ToString('yyyy-MM-dd'))
if (Test-Path $todayLog) {
    $lastLine = Get-Content $todayLog -Tail 1 -EA SilentlyContinue
    if ($lastLine -match '^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})') {
        $lastAct = [datetime]::Parse($Matches[1])
        $c = if (($now - $lastAct).TotalMinutes -gt 15) { 'Yellow' } else { 'Gray' }
        Write-Finding "Last app-log entry:" "$($lastAct.ToString('yyyy-MM-dd HH:mm:ss'))  ($(Format-Delta ($now - $lastAct)) ago)" $c
        Write-Host "    (app log only writes on AppStore changes, so this is a floor on" -ForegroundColor DarkGray
        Write-Host "     'last alive', not a precise freeze time)" -ForegroundColor DarkGray
    }
}

# Determine the freeze anchor to correlate against
if ($FreezeTime) {
    try { $tdFreezeAnchor = [datetime]::Parse($FreezeTime) } catch { Write-Host "    (could not parse -FreezeTime '$FreezeTime')" -ForegroundColor Yellow }
} elseif ($tdFrozenNow) {
    $tdFreezeAnchor = $now
}

# -- Timeline --------------------------------------------------------------

Write-Section "SLEEP / WAKE TIMELINE  (last ${HoursBack}h)"

if (-not $sorted -or $sorted.Count -eq 0) {
    Write-Host "    No sleep/wake/standby events in the scan window." -ForegroundColor DarkGray
    Write-Host "    (If TD still froze, the trigger is likely NOT sleep -- look at" -ForegroundColor DarkGray
    Write-Host "     display-DPMS or the file-sync deadlock path instead.)" -ForegroundColor DarkGray
} else {
    Write-Host "    Time                  Event         Detail" -ForegroundColor DarkGray
    $prev = $null
    foreach ($ev in $sorted) {
        # show gap since previous event if it's a wake following a sleep
        $line = "    {0}  {1,-12}  {2}" -f $ev.Time.ToString('yyyy-MM-dd HH:mm:ss'), $ev.Kind, $ev.Detail
        Write-Host $line -ForegroundColor $ev.Color
        $prev = $ev
    }

    # Pair sleep->wake durations
    Write-Section "SLEEP SESSIONS  (paired)"
    $sleeps = $sorted | Where-Object { $_.Kind -in 'SLEEP','MS-ENTER' }
    $wakes  = $sorted | Where-Object { $_.Kind -in 'RESUME','MS-EXIT' }
    if ($sleeps.Count -eq 0) {
        Write-Host "    No sleep entries to pair." -ForegroundColor DarkGray
    } else {
        foreach ($s in $sleeps) {
            $w = $wakes | Where-Object { $_.Time -gt $s.Time } | Sort-Object Time | Select-Object -First 1
            if ($w) {
                $dur = $w.Time - $s.Time
                Write-Host ("    {0}  ->  {1}   ({2} asleep)" -f $s.Time.ToString('MM-dd HH:mm:ss'), $w.Time.ToString('HH:mm:ss'), (Format-Delta $dur)) -ForegroundColor Gray
            } else {
                Write-Host ("    {0}  ->  (no matching wake -- still asleep or crashed while asleep)" -f $s.Time.ToString('MM-dd HH:mm:ss')) -ForegroundColor Yellow
            }
        }
    }
}

# -- TD hang / crash WER events --------------------------------------------

Write-Section "TD HANG / CRASH EVENTS  (WER, last ${HoursBack}h)"

$werHang  = Get-WinEvent -FilterHashtable @{LogName='Application'; Id=1002; StartTime=$scanStart} -MaxEvents 200 -EA SilentlyContinue | Where-Object { $_.Message -match 'TouchDesigner' }
$werCrash = Get-WinEvent -FilterHashtable @{LogName='Application'; Id=1000; StartTime=$scanStart} -MaxEvents 200 -EA SilentlyContinue | Where-Object { $_.Message -match 'TouchDesigner' }
$werAll = @()
foreach ($e in $werHang)  { $werAll += [PSCustomObject]@{ Time = $e.TimeCreated; Kind = 'HANG (AppHangB1)' } }
foreach ($e in $werCrash) { $werAll += [PSCustomObject]@{ Time = $e.TimeCreated; Kind = 'CRASH (APPCRASH)' } }
$werAll = $werAll | Sort-Object Time

if ($werAll.Count -eq 0) {
    Write-Host "    None recorded. (Silent freezes often leave NO WER entry -- that's" -ForegroundColor DarkGray
    Write-Host "     the documented TD-internal-deadlock signature, so absence here does" -ForegroundColor DarkGray
    Write-Host "     not mean TD didn't freeze.)" -ForegroundColor DarkGray
} else {
    foreach ($w in $werAll) {
        Write-Host ("    {0}   {1}" -f $w.Time.ToString('yyyy-MM-dd HH:mm:ss'), $w.Kind) -ForegroundColor Red
    }
}

# -- Correlation -----------------------------------------------------------

Write-Section "CORRELATION"

$wakeEvents = $sorted | Where-Object { $_.Kind -in 'RESUME','MS-EXIT','WAKE-DETAIL' }

function Correlate-Anchor {
    param([datetime]$Anchor, [string]$Label)
    $priorWake  = $wakeEvents | Where-Object { $_.Time -le $Anchor } | Sort-Object Time | Select-Object -Last 1
    $priorSleep = ($sorted | Where-Object { $_.Kind -in 'SLEEP','MS-ENTER' -and $_.Time -le $Anchor } | Sort-Object Time | Select-Object -Last 1)
    Write-Host "    $Label @ $($Anchor.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor White
    if ($priorWake) {
        $d = $Anchor - $priorWake.Time
        $c = if ($d.TotalMinutes -le 10) { 'Red' } else { 'Gray' }
        Write-Host ("      nearest preceding wake:  {0}  ({1} before)" -f $priorWake.Time.ToString('HH:mm:ss'), (Format-Delta $d)) -ForegroundColor $c
        if ($d.TotalMinutes -le 10) {
            Write-Host "      *** freeze within 10 min of a wake -- strong sleep/wake link ***" -ForegroundColor Red
        }
    } else {
        Write-Host "      no wake event precedes this freeze -> sleep is probably NOT the trigger" -ForegroundColor Yellow
    }
    if ($priorSleep) {
        Write-Host ("      last sleep entry before it: {0} ({1})" -f $priorSleep.Time.ToString('HH:mm:ss'), $priorSleep.Kind) -ForegroundColor DarkGray
    }
}

$correlated = $false
if ($tdFreezeAnchor) {
    Correlate-Anchor $tdFreezeAnchor $(if ($tdFrozenNow -and -not $FreezeTime) { 'CURRENT FREEZE' } else { 'FREEZE' })
    $correlated = $true
}
foreach ($w in $werAll) {
    Correlate-Anchor $w.Time $w.Kind
    $correlated = $true
}
if (-not $correlated) {
    Write-Host "    No freeze anchor to correlate (TD is responding and no WER hangs)." -ForegroundColor DarkGray
    Write-Host "    Re-run with -FreezeTime once you catch a freeze, or leave the flight" -ForegroundColor DarkGray
    Write-Host "    recorder running so the next one is timestamped." -ForegroundColor DarkGray
}

Write-Host ""
