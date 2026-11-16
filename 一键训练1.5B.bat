@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo ==========================================================
echo   嘉桦1.1  1.5B 训练（下载后点这个就行）
echo   流程：建词表 -^> 预训练 -^> 问答微调
echo ==========================================================
py -3.13 -c "import torch" 2>nul
if errorlevel 1 (
    echo [错误] 当前 Python 没装 PyTorch，请先运行下面这行：
    echo         py -3.13 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    pause
    exit /b 1
)
py -3.13 -c "import torch,sys; sys.exit(0 if (torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory>=20*1024**3) else 1)"
if errorlevel 1 (
    echo.
    echo [提示] 没检测到 20GB 以上显存的显卡。
    echo        1.5B 全参数训练要 24GB 显存左右，8G 内存 / 1G 显存只能做推理，训不动。
    echo        选项：直接回车 = 继续（先给你看 1.5B 参数量和显存估算，到训练那一步会报错退出）
    echo              输入 1   = 改成小模型试跑（几分钟跑通整条流程）
    echo.
    set /p CHOICE=请输入 1 走小模型试跑，直接回车继续：
    if "%CHOICE%"=="1" (
        call "一键试跑.bat"
        exit /b 0
    )
)
echo.
echo [1/4] 打印 1.5B 配置与参数量...
py -3.13 model.py --config configs/1.5b.json
echo.
echo [2/4] 建词表...
py -3.13 tokenizer.py build --data data/corpus.txt data/qa.jsonl --out tokenizer.json
echo.
echo [3/4] 预训练（1.5B，全参数）...
py -3.13 train.py --config configs/1.5b.json --data data/corpus.txt --out_dir out_1.5b --max_steps 20000 --batch_size 1 --grad_accum 16 --seq_len 1024 --save_every 500 --eval_every 500
echo.
echo [4/4] 问答微调...
py -3.13 sft.py --ckpt out_1.5b/ckpt_last.pt --data data/qa.jsonl --out_dir out_1.5b_sft --max_steps 3000 --batch_size 2 --seq_len 512 --lr 1e-4
echo.
echo 训练完成。接着双击 一键问答1.5B.bat 就能提问了。
pause
