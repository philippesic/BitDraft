param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^/[A-Za-z0-9._/-]+$')]
    [string] $KeeperScriptPath,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^/[A-Za-z0-9._/-]+$')]
    [string] $KeeperRunPath,
    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string] $WslDistribution = 'Ubuntu',
    [ValidatePattern('^[a-z][a-z0-9_-]*$')]
    [string] $WslRunUser = 'philip'
)

# Invoke on the registered Windows host through tmux MCP. The operator first
# authenticates the CPU-only keeper source and a fresh private Linux run path.
# The Linux keeper's metadata remains the authoritative birth/stop identity.
$ErrorActionPreference = 'Stop'
$WslExecutable = Join-Path $env:SystemRoot 'System32\wsl.exe'
$KeeperCommand = '"{0}" -d {1} -u {2} -- /usr/bin/python3 -B {3} --run-dir {4}' -f `
    $WslExecutable, $WslDistribution, $WslRunUser, $KeeperScriptPath, $KeeperRunPath
$KeeperCreation = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
    -Arguments @{ CommandLine = $KeeperCommand }
if ($KeeperCreation.ReturnValue -ne 0) {
    throw ('Keeper process creation failed: {0}' -f $KeeperCreation.ReturnValue)
}
$KeeperWindowsPid = [uint32] $KeeperCreation.ProcessId
$KeeperWindowsProcess = Get-CimInstance -ClassName Win32_Process `
    -Filter ('ProcessId = {0}' -f $KeeperWindowsPid)
if ($null -eq $KeeperWindowsProcess) {
    throw ('Created keeper process already absent: {0}' -f $KeeperWindowsPid)
}
[ordered] @{
    schema = 'wsl_interop_keeper_windows_launch_v1'
    observed_utc = [DateTime]::UtcNow.ToString('o')
    return_value = [uint32] $KeeperCreation.ReturnValue
    windows_pid = $KeeperWindowsPid
    creation_date = $KeeperWindowsProcess.CreationDate.ToUniversalTime().ToString('o')
    command_line = $KeeperWindowsProcess.CommandLine
    linux_keeper_source = $KeeperScriptPath
    linux_run_path = $KeeperRunPath
    gpu_queried = $false
    model_loaded = $false
} | ConvertTo-Json -Depth 4
