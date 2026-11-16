@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist "out_tiny_sft\ckpt_final.pt" (
    echo [提示] 还没有试跑版的权重，先帮你跑一遍一键试跑...
    call "一键试跑.bat"
)
py -3.13 chat.py --ckpt out_tiny_sft/ckpt_final.pt --quant8
pause
