@echo off
rem env.bat —— 自动找一个能用的 Python，放进变量 PY
rem 不再写死 py -3.13，避免别人电脑上没这个版本时全部静默失败
set "PY="
call :trypy py -3.13
call :trypy py -3.12
call :trypy py -3.11
call :trypy py -3.10
call :trypy py -3.9
call :trypy py -3
call :trypy python
call :trypy python3
call :trypy py
exit /b 0

:trypy
%* -c "import sys" >nul 2>&1
if not errorlevel 1 if not defined PY set "PY=%*"
exit /b 0
