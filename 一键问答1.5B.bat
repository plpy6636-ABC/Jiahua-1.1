@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist "out_1.5b_sft\ckpt_final.pt" (
    echo [提示] 还没训练出 1.5B 的权重，找不到 out_1.5b_sft\ckpt_final.pt
    echo        请先双击：一键训练1.5B.bat
    pause
    exit /b 1
)
echo ==========================================================
echo   嘉桦1.1  问答（int8 量化，8G 内存也能跑）
echo   中文英文都能问，输入 退出 结束
echo ==========================================================
py -3.13 chat.py --ckpt out_1.5b_sft/ckpt_final.pt --quant8
pause
