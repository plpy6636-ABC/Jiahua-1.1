@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
echo ==== 试跑：357 万参数小模型，几分钟跑完，不需要显卡 ====
call env.bat
if not defined PY goto nopy
%PY% -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo 没检测到 PyTorch，正在自动安装……
    %PY% -m pip install torch
)
%PY% -c "import torch" >nul 2>&1
if errorlevel 1 goto fail
%PY% tokenizer.py build --data data/corpus.txt data/qa.jsonl --out tokenizer.json
if errorlevel 1 goto fail
%PY% train.py --config configs/tiny.json --data data/corpus.txt --out_dir out_tiny --max_steps 500 --batch_size 8 --seq_len 160 --log_every 100 --eval_every 250
if errorlevel 1 goto fail
%PY% sft.py --ckpt out_tiny\ckpt_final.pt --data data/qa.jsonl --out_dir out_tiny_sft --max_steps 1500 --batch_size 8 --seq_len 112 --lr 2e-4
if errorlevel 1 goto fail
%PY% chat.py --ckpt out_tiny_sft\ckpt_final.pt --quant8 --temperature 0 --question "你好，你是谁？"
echo.
echo 试跑完成，权重在 out_tiny_sft\ckpt_final.pt
pause
exit /b 0

:nopy
echo [错误] 没找到可用的 Python。去 python.org 装 3.10 以上，第一页勾选 Add python.exe to PATH。
pause
exit /b 1

:fail
echo [错误] 上一步失败了，把上面的报错截图发出来。
pause
exit /b 1
