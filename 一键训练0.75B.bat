@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo ==========================================================
echo   0.75B 训练（嘉桦1.0 配置，需 NVIDIA 显卡）
echo   训练前请把大语料放进 data\corpus.txt
echo ==========================================================
py -3.13 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo [错误] 没有检测到可用显卡（CUDA），0.75B 在 CPU 上跑不完。
    echo        想先跑通流程请双击：一键试跑.bat
    pause
    exit /b 1
)
py -3.13 model.py --config configs/0.75b.json
if exist "tokenizer.json" del "tokenizer.json"
py -3.13 tokenizer.py build --data data/corpus.txt data/qa.jsonl --out tokenizer.json
py -3.13 train.py --config configs/0.75b.json --data data/corpus.txt --out_dir out_0.75b --max_steps 20000 --batch_size 4 --grad_accum 8 --save_every 500 --eval_every 500
py -3.13 sft.py --ckpt out_0.75b/ckpt_last.pt --data data/qa.jsonl --out_dir out_0.75b_sft --max_steps 2000 --batch_size 4 --seq_len 512
echo.
echo 训练完成，权重在 out_0.75b 和 out_0.75b_sft
pause
