@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
call env.bat
if not defined PY goto nopy

set "CKPT="
if exist out_1.5b_sft\ckpt_final.pt set "CKPT=out_1.5b_sft\ckpt_final.pt"
if not defined CKPT if exist out_0.75b_sft\ckpt_final.pt set "CKPT=out_0.75b_sft\ckpt_final.pt"
if not defined CKPT if exist out_tiny_sft\ckpt_final.pt set "CKPT=out_tiny_sft\ckpt_final.pt"
if not defined CKPT goto noweight

%PY% -c "import torch" >nul 2>&1
if errorlevel 1 goto notorch

echo 使用权重：%CKPT%
%PY% chat.py --ckpt "%CKPT%" --quant8
pause
exit /b 0

:nopy
echo [错误] 没找到可用的 Python。去 python.org 装 3.10 以上，第一页勾选 Add python.exe to PATH。
pause
exit /b 1

:notorch
echo [错误] 没装 PyTorch，正在自动安装……
%PY% -m pip install torch
echo 装完请重新双击本脚本。
pause
exit /b 1

:noweight
echo.
echo [提示] 还没训练过，所以没有权重，也就无法问答。
echo        权重不会随代码下载，必须自己训一次：
echo          · 有 20GB 显存：双击 一键训练1.5B.bat
echo          · 没显卡：双击 一键试跑.bat
pause
exit /b 1
