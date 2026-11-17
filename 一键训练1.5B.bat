@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
call env.bat
if not defined PY goto nopy
%PY% -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo 没检测到 PyTorch，正在自动安装……
    %PY% -m pip install torch
)
%PY% -c "import torch" >nul 2>&1
if errorlevel 1 goto fail

%PY% -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory/1024**3>=18 else 2)"
if errorlevel 2 goto nogpu

%PY% tokenizer.py build --data data/corpus.txt data/qa.jsonl --out tokenizer.json
if errorlevel 1 goto fail
%PY% train.py --config configs/1.5b.json --data data/corpus.txt --out_dir out_1.5b --max_steps 20000 --batch_size 1 --grad_accum 16 --seq_len 1024
if errorlevel 1 goto fail
%PY% sft.py --ckpt out_1.5b\ckpt_last.pt --data data/qa.jsonl --out_dir out_1.5b_sft --max_steps 3000 --batch_size 2 --seq_len 512 --lr 1e-4
if errorlevel 1 goto fail

echo.
echo 训练完成，权重在 out_1.5b_sft\ckpt_final.pt，接下来双击 一键问答1.5B.bat
pause
exit /b 0

:nopy
echo [错误] 没找到可用的 Python。去 python.org 装 3.10 以上，第一页勾选 Add python.exe to PATH。
pause
exit /b 1

:nogpu
echo.
echo [停止] 1.5B 全参数训练要 20GB 左右显存，这台电脑不够，没开始训练。
echo        有卡：直接双击 一键问答1.5B.bat 用别人训好的权重不行，权重必须自己训。
echo        没卡：双击 一键试跑.bat，用小模型验证整条流程。
pause
exit /b 1

:fail
echo [错误] 上一步失败了，把上面的报错截图发出来。
pause
exit /b 1
