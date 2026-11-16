@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo ==========================================================
echo   一键试跑：建词表 -^> 预训练 -^> 问答微调 -^> 自动提问
echo   （微小模型，笔记本 CPU 几分钟跑完）
echo ==========================================================
py -3.13 -c "import torch" 2>nul
if errorlevel 1 (
    echo [错误] 当前 Python 没装 PyTorch。
    echo        请先运行： py -3.13 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    pause
    exit /b 1
)
echo.
echo [1/4] 建词表...
py -3.13 tokenizer.py build --data data/corpus.txt data/qa.jsonl --out tokenizer.json
echo.
echo [2/4] 预训练 500 步...
py -3.13 train.py --config configs/tiny.json --data data/corpus.txt --out_dir out_tiny --max_steps 500 --batch_size 8 --seq_len 160 --eval_every 250
echo.
echo [3/4] 问答微调 1500 步...
py -3.13 sft.py --ckpt out_tiny/ckpt_final.pt --data data/qa.jsonl --out_dir out_tiny_sft --max_steps 1500 --batch_size 8 --seq_len 112 --lr 2e-4
echo.
echo [4/4] 提问测试（中文 + 英文）...
py -3.13 chat.py --ckpt out_tiny_sft/ckpt_final.pt --quant8 --question "你好，你是谁？"
py -3.13 chat.py --ckpt out_tiny_sft/ckpt_final.pt --quant8 --question "who are you"
echo.
echo 全部完成。想要 1.5B 正式版：双击 一键训练1.5B.bat
pause
