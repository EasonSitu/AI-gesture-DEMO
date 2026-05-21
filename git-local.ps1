$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$git = Join-Path $repo ".tools\PortableGit\cmd\git.exe"

if (-not (Test-Path $git)) {
    Write-Error "未找到便携版 Git：$git。请确认 .tools\PortableGit 存在。"
}

& $git -c safe.directory="$repo" @args

