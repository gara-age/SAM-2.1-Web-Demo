param(
    [Parameter(Mandatory=$true)][string]$LocalAddress,
    [Parameter(Mandatory=$true)][string]$RemoteSubnet,
    [ValidateRange(1,65535)][int]$Port=8000,
    [string]$RequestId='manual'
)
$ErrorActionPreference='Stop'
$resultFile=Join-Path $PSScriptRoot '.cache\lan-firewall-result.json'
try {
    $principal=New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Administrator permission is required.' }
    $address=[Net.IPAddress]::Parse($LocalAddress)
    if ($address.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetwork) { throw 'An IPv4 address is required.' }
    $subnetParts=$RemoteSubnet.Split('/')
    if ($subnetParts.Count -ne 2 -or [int]$subnetParts[1] -lt 1 -or [int]$subnetParts[1] -gt 32) { throw 'A specific IPv4 subnet is required.' }
    $null=[Net.IPAddress]::Parse($subnetParts[0])
    $interface=Get-NetIPAddress -AddressFamily IPv4 -IPAddress $LocalAddress | Select-Object -First 1
    if (-not $interface) { throw 'The requested local IPv4 address is not assigned to this PC.' }
    $ruleName='SAMVisionConsole-SchoolLAN-'+$Port
    $existing=Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue
    if ($existing) {
        Set-NetFirewallRule -Name $ruleName -Enabled True -Direction Inbound -Action Allow -Profile Any -EdgeTraversalPolicy Block -InterfaceAlias $interface.InterfaceAlias
        $existing | Get-NetFirewallPortFilter | Set-NetFirewallPortFilter -Protocol TCP -LocalPort $Port -RemotePort Any
        $existing | Get-NetFirewallAddressFilter | Set-NetFirewallAddressFilter -LocalAddress $LocalAddress -RemoteAddress $RemoteSubnet
    } else {
        New-NetFirewallRule -Name $ruleName -DisplayName ('SAM Vision Console - School LAN TCP '+$Port) `
            -Enabled True -Direction Inbound -Action Allow -Profile Any -Protocol TCP -LocalPort $Port `
            -LocalAddress $LocalAddress -RemoteAddress $RemoteSubnet -InterfaceAlias $interface.InterfaceAlias -EdgeTraversalPolicy Block | Out-Null
    }
    $rule=Get-NetFirewallRule -Name $ruleName
    $portFilter=$rule | Get-NetFirewallPortFilter
    $addressFilter=$rule | Get-NetFirewallAddressFilter
    $interfaceFilter=$rule | Get-NetFirewallInterfaceFilter
    $result=@{success=$true;request_id=$RequestId;rule_name=$ruleName;enabled=[string]$rule.Enabled;action=[string]$rule.Action;
        protocol=[string]$portFilter.Protocol;local_port=@($portFilter.LocalPort);local_address=@($addressFilter.LocalAddress);
        remote_address=@($addressFilter.RemoteAddress);interface_alias=@($interfaceFilter.InterfaceAlias);time=(Get-Date -Format o)}
    $result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $resultFile -Encoding utf8
} catch {
    @{success=$false;request_id=$RequestId;error=$_.Exception.Message;time=(Get-Date -Format o)} | ConvertTo-Json | Set-Content -LiteralPath $resultFile -Encoding utf8
    exit 1
}
