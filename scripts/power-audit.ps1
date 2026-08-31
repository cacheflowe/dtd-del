<#
.SYNOPSIS
    Power / sleep configuration audit -- one-shot snapshot of how this machine
    behaves when idle. Answers the question that matters for the "left it and
    came back to a frozen TD" bug: does this box use S0 Modern Standby, and
    when/why does it sleep?

    Companion to sleep-wake-correlator.ps1. Run this first to learn the
    machine's sleep policy; run the correlator to see what actually happened.

.DESCRIPTION
    Wraps the relevant built-in `powercfg` queries and interprets them for the
    Optimus / TouchDesigner freeze investigation documented in
    _assets/TD_CRASHES.md:
      - powercfg /a          -- which sleep states exist (S0 Modern Standby vs S3)
      - active plan timeouts -- sleep-after / display-off / hibernate (AC + DC)
      - powercfg /requests   -- what is currently blocking sleep
      - powercfg /lastwake   -- what last woke the machine
      - powercfg /sleepstudy -- full Modern Standby session report (HTML)

    Modern Standby (S0 Low Power Idle) is the prime suspect: on this Optimus
    chassis it powers the dGPU down while TD keeps running, TD's Vulkan device
    is lost without a TDR, and TD deadlocks on resume.

.PARAMETER SleepStudy
    Also generate a powercfg /sleepstudy HTML report (Modern Standby only).
    Written next to this script as sleepstudy-<timestamp>.html.

.PARAMETER Open
    Open the generated sleepstudy report in the default browser when done.
    Implies -SleepStudy.

.EXAMPLE
    .\power-audit.ps1
    .\power-audit.ps1 -SleepStudy
    .\power-audit.ps1 -Open
#>

param(
    [switch]$SleepStudy,
    [switch]$Open
)

if ($Open) { $SleepStudy = $true }

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

# Query a single powercfg setting on the active scheme and return AC/DC seconds.
function Get-PowerTimeout {
    param([string]$SubGroup, [string]$Setting)
    $out = powercfg /query SCHEME_CURRENT $SubGroup $Setting 2>$null
    if (-not $out) { return $null }
    $ac = $null; $dc = $null
    foreach ($line in $out) {
        if ($line -match 'Current AC Power Setting Index:\s*0x([0-9a-fA-F]+)') { $ac = [Convert]::ToInt32($Matches[1], 16) }
        if ($line -match 'Current DC Power Setting Index:\s*0x([0-9a-fA-F]+)') { $dc = [Convert]::ToInt32($Matches[1], 16) }
    }
    [PSCustomObject]@{ AC = $ac; DC = $dc }
}

function Format-Timeout {
    param([int]$Seconds)
    if ($null -eq $Seconds) { return '?' }
    if ($Seconds -eq 0) { return 'Never' }
    if ($Seconds -lt 60) { return "${Seconds}s" }
    if ($Seconds -lt 3600) { return "$([math]::Round($Seconds / 60))m" }
    return "$([math]::Round($Seconds / 3600, 1))h"
}

# -- Header ----------------------------------------------------------------

Clear-Host
Write-Host "  POWER AUDIT " -BackgroundColor DarkBlue -ForegroundColor White -NoNewline
Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor DarkGray

# -- Available sleep states ------------------------------------------------

Write-Section "AVAILABLE SLEEP STATES  (powercfg /a)"

$avail = powercfg /a 2>$null | Out-String
$modernStandby = $avail -match 'Standby \(S0 Low Power Idle\)'
$s3 = $avail -match 'Standby \(S3\)'
$hibernate = $avail -match 'Hibernate'

if ($modernStandby) {
    Write-Finding "Modern Standby (S0):" "PRESENT  <-- prime suspect for the idle freeze" 'Red'
} else {
    Write-Finding "Modern Standby (S0):" "not present" 'Green'
}
Write-Finding "Classic Sleep (S3):" $(if ($s3) { 'available' } else { 'not available' }) 'Gray'
Write-Finding "Hibernate:" $(if ($hibernate) { 'available' } else { 'not available' }) 'Gray'

# Show the raw list too (it also explains WHY unavailable states are disabled)
Write-Host ""
foreach ($line in ($avail -split "`r?`n")) {
    $t = $line.Trim()
    if (-not $t) { continue }
    $c = 'DarkGray'
    if ($t -match 'S0 Low Power Idle') { $c = 'Yellow' }
    elseif ($t -match '^The following sleep states are available') { $c = 'Gray' }
    elseif ($t -match 'are not available') { $c = 'DarkGray' }
    Write-Host "    $t" -ForegroundColor $c
}

# -- Active plan --------------------------------------------------------

Write-Section "ACTIVE POWER PLAN TIMEOUTS"

$active = powercfg /getactivescheme 2>$null | Out-String
if ($active -match 'Power Scheme GUID:\s*[0-9a-fA-F-]+\s*\((.+?)\)') {
    Write-Finding "Active plan:" $Matches[1].Trim() 'Cyan'
}

$sleepTo   = Get-PowerTimeout 'SUB_SLEEP' 'STANDBYIDLE'
$displayTo = Get-PowerTimeout 'SUB_VIDEO' 'VIDEOIDLE'
$hiberTo   = Get-PowerTimeout 'SUB_SLEEP' 'HIBERNATEIDLE'

if ($sleepTo) {
    $c = if ($sleepTo.AC -eq 0) { 'Green' } else { 'Yellow' }
    Write-Color "    Sleep after (AC): "; Write-Color (Format-Timeout $sleepTo.AC) $c
    Write-Host "   (DC: $(Format-Timeout $sleepTo.DC))" -ForegroundColor DarkGray
}
if ($displayTo) {
    Write-Color "    Display off (AC): "; Write-Color (Format-Timeout $displayTo.AC) 'Gray'
    Write-Host "   (DC: $(Format-Timeout $displayTo.DC))" -ForegroundColor DarkGray
}
if ($hiberTo) {
    Write-Color "    Hibernate (AC):   "; Write-Color (Format-Timeout $hiberTo.AC) 'Gray'
    Write-Host "   (DC: $(Format-Timeout $hiberTo.DC))" -ForegroundColor DarkGray
}

# -- Sleep blockers --------------------------------------------------------

Write-Section "CURRENT SLEEP BLOCKERS  (powercfg /requests)"

$requests = powercfg /requests 2>$null
$anyRequest = $false
$curCat = $null
foreach ($line in $requests) {
    $t = $line.Trim()
    if (-not $t) { continue }
    if ($t -match '^(DISPLAY|SYSTEM|AWAYMODE|EXECUTION|PERFBOOST|ACTIVELOCKSCREEN):') {
        $curCat = $t.TrimEnd(':')
        continue
    }
    if ($t -eq 'None.') { continue }
    $anyRequest = $true
    $c = if ($t -match 'TouchDesigner') { 'Yellow' } else { 'Gray' }
    Write-Host "    [$curCat] $t" -ForegroundColor $c
}
if (-not $anyRequest) {
    Write-Finding "Blockers:" "None -- nothing is preventing sleep right now" 'Green'
} else {
    Write-Host "    (an active SYSTEM/EXECUTION request keeps the box awake; its" -ForegroundColor DarkGray
    Write-Host "     absence means TD does NOT hold sleep off while running)" -ForegroundColor DarkGray
}

# -- Last wake -------------------------------------------------------------

Write-Section "LAST WAKE  (powercfg /lastwake)"

$lastwake = powercfg /lastwake 2>$null
foreach ($line in $lastwake) {
    $t = $line.Trim()
    if (-not $t -or $t -match '^Wake History Count') { continue }
    Write-Host "    $t" -ForegroundColor Gray
}

# -- Sleep study -----------------------------------------------------------

if ($SleepStudy) {
    Write-Section "SLEEP STUDY REPORT"
    if (-not $modernStandby) {
        Write-Finding "Skipped:" "sleepstudy only works on Modern Standby systems" 'DarkGray'
    } else {
        $stamp = Get-Date -Format 'yyyy-MM-dd_HHmmss'
        $outPath = Join-Path $PSScriptRoot "sleepstudy-$stamp.html"
        powercfg /sleepstudy /output "$outPath" 2>$null | Out-Null
        if (Test-Path $outPath) {
            Write-Finding "Report:" $outPath 'Cyan'
            Write-Host "    Open it to see every Modern Standby session, its duration," -ForegroundColor DarkGray
            Write-Host "    battery drain, and which component kept the SoC awake." -ForegroundColor DarkGray
            if ($Open) { Start-Process $outPath }
        } else {
            Write-Finding "Report:" "generation failed (needs admin?)" 'Yellow'
        }
    }
}

# -- Verdict ---------------------------------------------------------------

Write-Section "READING"

if ($modernStandby) {
    Write-Host "    This machine uses S0 Modern Standby. When you walk away it does" -ForegroundColor Yellow
    Write-Host "    NOT do a clean S3 sleep -- it stays in a low-power state where the" -ForegroundColor Yellow
    Write-Host "    dGPU can be powered down while TD is still 'running'. That matches" -ForegroundColor Yellow
    Write-Host "    the frozen-on-return signature (Vulkan device lost, no TDR)." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "    Next: run sleep-wake-correlator.ps1 to confirm a sleep/wake event" -ForegroundColor Gray
    Write-Host "    lines up with the freeze. If it does, the A/B test is to disable" -ForegroundColor Gray
    Write-Host "    system sleep (keep display-off on) and see if freezes stop:" -ForegroundColor Gray
    Write-Host "        powercfg /change standby-timeout-ac 0" -ForegroundColor DarkCyan
} else {
    Write-Host "    No Modern Standby -- if 'Sleep after (AC)' above is a real timeout" -ForegroundColor Gray
    Write-Host "    (not Never), classic S3 sleep is still a candidate. If it's Never" -ForegroundColor Gray
    Write-Host "    and display-off is the only idle action, the trigger is more likely" -ForegroundColor Gray
    Write-Host "    display-DPMS / Optimus copy-back re-init than sleep." -ForegroundColor Gray
}
Write-Host ""
