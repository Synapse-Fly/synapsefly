<#
.SYNOPSIS
    Start the FlyBrain backend (:4000) and the Next frontend (:3000) as two PowerShell jobs.

.DESCRIPTION
    SPEC h.4 / i.5. Sets $env:PYTHONUTF8 = "1", starts "py -3 backend\run.py" as a PowerShell job
    (merged stdout+stderr to out\backend.log) and "npm run dev" in frontend\ as a second job
    (out\frontend.log), waits up to -TimeoutSeconds for GET /api/health to answer "ok":true, opens
    http://localhost:3000 and then tails both logs until Ctrl+C, which stops both jobs (and the
    processes they started).

    Never uses --reload: uvicorn's reloader spawns a second process and the simulation would start twice.

.PARAMETER NoFrontend
    Backend only (use your own "npm run dev" in frontend\).

.PARAMETER Port
    Backend port. Default: $env:FLY_PORT, then FLY_PORT from .env, then 4000.

.PARAMETER FrontendPort
    Frontend port passed to "npm run dev -- --port". Default 3000.

.PARAMETER TimeoutSeconds
    How long to wait for /api/health. Default 60 (SPEC h.4).

.PARAMETER NoBrowser
    Do not open a browser window.

.PARAMETER Help
    Print this usage summary and exit 0 (SPEC h.4: every script accepts -h).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\dev.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\dev.ps1 -NoFrontend -Port 4001
#>
[CmdletBinding()]
param(
    [switch] $NoFrontend,
    [int]    $Port = 0,
    [int]    $FrontendPort = 3000,
    [int]    $TimeoutSeconds = 60,
    [switch] $NoBrowser,
    [Alias('h')]
    [switch] $Help
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

# ---------------------------------------------------------------------------- -h (SPEC h.4)
if ($Help) {
    @(
        'dev.ps1 -- start the FlyBrain backend (:4000) and the Next frontend (:3000) as two jobs.',
        '',
        'Usage:',
        '  powershell -ExecutionPolicy Bypass -File scripts\dev.ps1 [-NoFrontend] [-Port <int>]',
        '             [-FrontendPort <int>] [-TimeoutSeconds <int>] [-NoBrowser] [-h|-Help]',
        '',
        'Options:',
        '  -NoFrontend        backend only (run "npm run dev" in frontend\ yourself)',
        '  -Port <int>        backend port; default $env:FLY_PORT, then FLY_PORT from .env, then 4000',
        '  -FrontendPort <n>  port passed to "npm run dev -- --port"; default 3000',
        '  -TimeoutSeconds    how long to wait for GET /api/health to report ok:true; default 60',
        '  -NoBrowser         do not open a browser window',
        '  -h, -Help          print this and exit 0',
        '',
        'What it does: sets PYTHONUTF8=1, starts "py -3 backend\run.py" (out\backend.log) and',
        '"npm run dev" in frontend\ (out\frontend.log) as PowerShell jobs, waits for',
        'GET /api/health to report ok:true, opens http://localhost:3000 and tails both logs.',
        'Ctrl+C stops both jobs and the processes they started. Never uses --reload.',
        '',
        'Full help: Get-Help -Full scripts\dev.ps1'
    ) | ForEach-Object { Write-Host $_ }
    exit 0
}

# ---------------------------------------------------------------------------- paths (never CWD-dependent)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo      = Split-Path -Parent $ScriptDir
$OutDir    = Join-Path $Repo 'out'
$BackendLog  = Join-Path $OutDir 'backend.log'
$FrontendLog = Join-Path $OutDir 'frontend.log'
$PidDir    = Join-Path $OutDir 'dev'

$script:Jobs = @()      # @{ Name; Job; PidFile; Log }

function Write-Head([string] $text) { Write-Host ('=== ' + $text) -ForegroundColor Cyan }
function Write-Warn([string] $text) { Write-Host ('WARN: ' + $text) -ForegroundColor Yellow }
function Write-Err ([string] $text) { Write-Host ('ERROR: ' + $text) -ForegroundColor Red }

function Get-DotEnvValue([string] $key) {
    $envFile = Join-Path $Repo '.env'
    if (-not (Test-Path -LiteralPath $envFile)) { return '' }
    foreach ($line in Get-Content -LiteralPath $envFile -Encoding UTF8) {
        $s = $line.Trim()
        if ($s.Length -eq 0) { continue }
        if ($s.StartsWith('#')) { continue }
        $i = $s.IndexOf('=')
        if ($i -lt 1) { continue }
        if ($s.Substring(0, $i).Trim() -eq $key) { return $s.Substring($i + 1).Trim() }
    }
    return ''
}

function Resolve-Port {
    if ($Port -gt 0) { return $Port }
    if ($env:FLY_PORT) {
        $parsed = 0
        if ([int]::TryParse($env:FLY_PORT, [ref] $parsed)) { if ($parsed -gt 0) { return $parsed } }
    }
    $fromEnvFile = Get-DotEnvValue 'FLY_PORT'
    if ($fromEnvFile) {
        $parsed = 0
        if ([int]::TryParse($fromEnvFile, [ref] $parsed)) { if ($parsed -gt 0) { return $parsed } }
    }
    return 4000
}

# ---------------------------------------------------------------------------- jobs
# The job starts the real process through cmd.exe so that stdout AND stderr land merged in one log file
# (Windows PowerShell 5.1 turns a native command's stderr into ErrorRecords, which would mark the job
# failed on perfectly normal uvicorn output). The child PID is written to a file so Stop-Dev can kill the
# whole tree -- stopping the job alone would orphan python.exe / node.exe and leave the port bound.
function Start-DevJob([string] $name, [string] $workDir, [string] $commandLine, [string] $log) {
    $pidFile = Join-Path $PidDir ($name + '.pid')
    if (Test-Path -LiteralPath $pidFile) { Remove-Item -LiteralPath $pidFile -Force }
    if (Test-Path -LiteralPath $log) { Remove-Item -LiteralPath $log -Force }
    $job = Start-Job -Name ('flybrain-' + $name) -ScriptBlock {
        param($workDir, $commandLine, $log, $pidFile)
        $env:PYTHONUTF8 = '1'
        $env:PYTHONIOENCODING = 'utf-8'
        Set-Location -LiteralPath $workDir
        # cmd /s /c strips the outermost quotes and runs the rest verbatim, so stdout and stderr end up
        # merged in ONE log file without PowerShell re-quoting or wrapping a native command's stderr in
        # ErrorRecords. ProcessStartInfo (not Start-Process -NoNewWindow, which hangs in a job that has
        # no console) with CreateNoWindow: no flashing console, and a real PID for the cleanup.
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $env:ComSpec
        $psi.Arguments = '/d /s /c "' + $commandLine + ' > "' + $log + '" 2>&1"'
        $psi.WorkingDirectory = $workDir
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        $proc = [System.Diagnostics.Process]::Start($psi)
        Set-Content -LiteralPath $pidFile -Value $proc.Id -Encoding ascii
        $proc.WaitForExit()
        $code = 0
        try { $code = [int] $proc.ExitCode } catch { $code = 0 }
        exit $code
    } -ArgumentList $workDir, $commandLine, $log, $pidFile
    $script:Jobs += @{ Name = $name; Job = $job; PidFile = $pidFile; Log = $log }
    return $job
}

function Stop-Dev {
    Write-Host ''
    Write-Head 'stopping'
    foreach ($entry in $script:Jobs) {
        if (Test-Path -LiteralPath $entry.PidFile) {
            $raw = (Get-Content -LiteralPath $entry.PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
            $childPid = 0
            if ($raw -and [int]::TryParse($raw.ToString().Trim(), [ref] $childPid)) {
                if ($childPid -gt 0) {
                    Write-Host ("  killing $($entry.Name) process tree (pid $childPid)")
                    & taskkill /PID $childPid /T /F 2>$null | Out-Null
                }
            }
            Remove-Item -LiteralPath $entry.PidFile -Force -ErrorAction SilentlyContinue
        }
        try { Stop-Job -Job $entry.Job -ErrorAction SilentlyContinue } catch { }
        try { Remove-Job -Job $entry.Job -Force -ErrorAction SilentlyContinue } catch { }
        Write-Host ("  $($entry.Name) stopped; log: $($entry.Log)")
    }
    $script:Jobs = @()
}

function Test-Health([int] $port) {
    $url = 'http://127.0.0.1:' + $port + '/api/health'
    try {
        $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3
    } catch {
        return $null
    }
    if ($resp.StatusCode -ne 200) { return $null }
    try { return ($resp.Content | ConvertFrom-Json) } catch { return $null }
}

# SPEC h.4: wait for /api/health to be *ok*, not merely to answer. A 200 with "ok": false means the
# simulation has not come up (no tick yet, a failed lifespan step), and proceeding would open a browser
# on a dead brain. StrictMode 2.0 forbids touching a property that is not there, hence the lookup.
function Test-HealthOk($health) {
    if ($null -eq $health) { return $false }
    if ($health.PSObject.Properties.Name -notcontains 'ok') { return $false }
    return ($health.ok -eq $true)
}

function Format-Health($health) {
    if ($null -eq $health) { return '(no answer)' }
    $text = '(unprintable)'
    try { $text = ($health | ConvertTo-Json -Depth 2 -Compress) } catch { }
    $text = [string] $text
    $text = $text -creplace '[^\x09\x20-\x7E]', '?'      # ASCII-only console (SPEC 0.1)
    if ($text.Length -gt 400) { $text = $text.Substring(0, 400) + '...' }
    return $text
}

function Show-NewLines([hashtable] $entry, [hashtable] $offsets, [bool] $flush = $false) {
    if (-not (Test-Path -LiteralPath $entry.Log)) { return }
    # UTF-8: both services write UTF-8 (PYTHONUTF8=1 / node). Non-ASCII is replaced by '?' so a cp1254
    # console never garbles or throws (SPEC 0.1: ASCII-only console output).
    $all = @(Get-Content -LiteralPath $entry.Log -Encoding UTF8 -ErrorAction SilentlyContinue)
    $seen = 0
    if ($offsets.ContainsKey($entry.Name)) { $seen = [int] $offsets[$entry.Name] }
    if ($all.Count -lt $seen) { $seen = 0 }                 # log was truncated / rotated
    # the writer may be mid-line: hold the last line back until more arrives (or until the final flush)
    $upto = $all.Count
    if (-not $flush) { $upto = $all.Count - 1 }
    if ($upto -le $seen) {
        $offsets[$entry.Name] = $seen
        return
    }
    for ($i = $seen; $i -lt $upto; $i++) {
        $line = [string] $all[$i]
        # -creplace, never -replace: the case-insensitive operator applies Turkish casing on a tr-TR
        # machine, where 'I' lowercases to the dotless 'i' and falls outside [\x20-\x7E] -> "?NFO"
        $line = $line -creplace "$([char]27)\[[0-9;?]*[A-Za-z]", ''   # ANSI colour / cursor sequences
        $line = $line -creplace '[^\x09\x20-\x7E]', '?'
        Write-Host ('[' + $entry.Name + '] ' + $line)
    }
    $offsets[$entry.Name] = $upto
}

# ---------------------------------------------------------------------------- preflight
$BackendPort = Resolve-Port
$BackendUrl  = 'http://127.0.0.1:' + $BackendPort
$FrontendUrl = 'http://localhost:' + $FrontendPort

Write-Head 'FlyBrain dev'
Write-Host ("repo      $Repo")
$env:PYTHONUTF8 = '1'
Write-Host 'PYTHONUTF8 = 1'

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path $PidDir | Out-Null

$runPy = Join-Path $Repo 'backend\run.py'
if (-not (Test-Path -LiteralPath $runPy)) {
    Write-Err "backend\run.py not found at $runPy"
    exit 2
}
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Err "'py' is not on PATH. Install Python 3.12+ with the py launcher."
    exit 2
}

$envFile = Join-Path $Repo '.env'
$envExample = Join-Path $Repo '.env.example'
if (-not (Test-Path -LiteralPath $envFile)) {
    if (Test-Path -LiteralPath $envExample) {
        Write-Warn ".env is missing; copying .env.example (every default runs offline)"
        Copy-Item -LiteralPath $envExample -Destination $envFile
    } else {
        Write-Warn ".env is missing and there is no .env.example; using the built-in defaults"
    }
}

$doFrontend = -not $NoFrontend
if ($doFrontend) {
    $pkg = Join-Path $Repo 'frontend\package.json'
    if (-not (Test-Path -LiteralPath $pkg)) {
        Write-Warn "frontend\package.json not found; running the backend only"
        $doFrontend = $false
    } elseif (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        Write-Warn "'npm' is not on PATH; running the backend only"
        $doFrontend = $false
    } elseif (-not (Test-Path -LiteralPath (Join-Path $Repo 'frontend\node_modules'))) {
        Write-Warn "frontend\node_modules is missing; run 'npm install' in frontend\ first. Backend only."
        $doFrontend = $false
    }
    if ($doFrontend) {
        $localEnv = Join-Path $Repo 'frontend\.env.local'
        $localExample = Join-Path $Repo 'frontend\.env.local.example'
        if ((-not (Test-Path -LiteralPath $localEnv)) -and (Test-Path -LiteralPath $localExample)) {
            Write-Warn "frontend\.env.local is missing; copying frontend\.env.local.example"
            Copy-Item -LiteralPath $localExample -Destination $localEnv
        }
    }
}

# ---------------------------------------------------------------------------- run
$exitCode = 0
try {
    Write-Head 'starting jobs'
    Start-DevJob 'backend' $Repo 'py -3 backend\run.py' $BackendLog | Out-Null
    Write-Host ("  backend  : py -3 backend\run.py   (no --reload)  -> $BackendLog")
    if ($doFrontend) {
        $frontDir = Join-Path $Repo 'frontend'
        Start-DevJob 'frontend' $frontDir ('npm run dev -- --port ' + $FrontendPort) $FrontendLog | Out-Null
        Write-Host ("  frontend : npm run dev -- --port $FrontendPort  -> $FrontendLog")
    }

    Write-Head "waiting for $BackendUrl/api/health to report ok:true (up to $TimeoutSeconds s)"
    $health = $null
    $healthOk = $false
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $offsets = @{}
    while ((Get-Date) -lt $deadline) {
        foreach ($entry in $script:Jobs) { Show-NewLines $entry $offsets }
        $backendJob = ($script:Jobs | Where-Object { $_.Name -eq 'backend' } | Select-Object -First 1).Job
        if ($backendJob.State -eq 'Completed' -or $backendJob.State -eq 'Failed') {
            Write-Err "the backend job exited before /api/health answered; last lines of $BackendLog :"
            if (Test-Path -LiteralPath $BackendLog) {
                Get-Content -LiteralPath $BackendLog -Tail 30 | ForEach-Object { Write-Host ('  ' + $_) }
            }
            $exitCode = 1
            break
        }
        $answer = Test-Health $BackendPort
        if ($answer) { $health = $answer }          # keep the last body, ok or not, for the timeout report
        if (Test-HealthOk $answer) { $healthOk = $true; break }
        Start-Sleep -Milliseconds 500
    }

    if ($exitCode -ne 0) { exit $exitCode }
    if (-not $healthOk) {
        if ($health) {
            Write-Err ("$BackendUrl/api/health answered but never reported ok:true within " +
                       "$TimeoutSeconds s (see $BackendLog). Last body: " + (Format-Health $health))
        } else {
            Write-Err "no answer from $BackendUrl/api/health within $TimeoutSeconds s (see $BackendLog)"
        }
        $exitCode = 1
        exit $exitCode
    }

    $okText = 'unknown'
    if ($health.PSObject.Properties.Name -contains 'ok') { $okText = [string] $health.ok }
    $rtfText = 'n/a'
    if ($health.PSObject.Properties.Name -contains 'rtf') {
        # invariant culture: a Turkish console would otherwise print "1,11"
        $rtfText = [string]::Format([System.Globalization.CultureInfo]::InvariantCulture,
                                    '{0:0.00}', [double] $health.rtf)
    }
    $runText = 'n/a'
    if ($health.PSObject.Properties.Name -contains 'run_id') { $runText = [string] $health.run_id }
    Write-Head 'up'
    Write-Host ("  health   : ok=$okText  rtf=$rtfText  run_id=$runText")
    Write-Host ("  backend  : $BackendUrl        (REST: $BackendUrl/api/health, WS: ws://127.0.0.1:$BackendPort/ws)")
    if ($doFrontend) {
        Write-Host ("  frontend : $FrontendUrl")
    } else {
        Write-Host '  frontend : not started (-NoFrontend)'
    }
    Write-Host ("  logs     : $BackendLog" + $(if ($doFrontend) { "  and  $FrontendLog" } else { '' }))
    Write-Host '  stop     : press Ctrl+C here (both jobs are stopped), or in another shell:' -ForegroundColor Green
    Write-Host '             Get-Job flybrain-* | Stop-Job; Get-Job flybrain-* | Remove-Job' -ForegroundColor Green

    if ($doFrontend -and -not $NoBrowser) {
        Write-Host ("opening $FrontendUrl")
        try { Start-Process $FrontendUrl | Out-Null } catch { Write-Warn "could not open a browser: $_" }
    }

    Write-Head 'tailing logs (Ctrl+C to stop both jobs)'
    while ($true) {
        $alive = $false
        foreach ($entry in $script:Jobs) {
            Show-NewLines $entry $offsets
            if ($entry.Job.State -eq 'Running' -or $entry.Job.State -eq 'NotStarted') { $alive = $true }
        }
        if (-not $alive) {
            foreach ($entry in $script:Jobs) { Show-NewLines $entry $offsets $true }
            Write-Warn 'every job has exited'
            break
        }
        Start-Sleep -Milliseconds 500
    }
} finally {
    Stop-Dev
}

exit $exitCode
