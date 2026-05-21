@echo off
setlocal
set "REPO=%~dp0"
set "GIT=%REPO%.tools\PortableGit\cmd\git.exe"

if not exist "%GIT%" (
  echo 未找到便携版 Git: %GIT%
  exit /b 1
)

"%GIT%" -c safe.directory="%REPO:~0,-1%" %*

