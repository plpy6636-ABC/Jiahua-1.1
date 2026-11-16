# -*- coding: utf-8 -*-
"""
sft.py —— 问答微调（让模型学会"用户问 -> 助手答"的格式和内容）

用法：
    py -3.13 sft.py --ckpt out_1.5b/ckpt_last.pt --data data/qa.jsonl --out_dir out_1.5b_sft --max_steps 3000 --batch_size 2 --seq_len 512

只在"回答"那一段算损失，问题部分不学（否则模型会学着重复你的问题）。
"""
import argparse
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from data import QADataset, collate_pad
from model import GPT, GPTConfig
from tokenizer import CharTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out_1.5b/ckpt_last.pt", help="预训练得到的权重")
    ap.add_argument("--data", default="data/qa.jsonl")
    ap.add_argument("--out_dir", default="out_1.5b_sft")
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 从权重里恢复配置和词表 ----
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = GPTConfig.from_dict(ck["config"])
    tok = CharTokenizer(ck["tokenizer_chars"])
    tok.save(os.path.join(args.out_dir, "tokenizer.json"))
    print(f"载入 {args.ckpt}：{cfg.n_layer} 层 / {cfg.d_model} 维 / 词表 {cfg.vocab_size} / 已完成 {ck['step']} 步")

    model = GPT(cfg)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        print("[提示] 权重不完全匹配：", missing, unexpected)
    print(f"参数量：{model.num_params():,}（{model.num_params() / 1e9:.3f} B）")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = device.type == "cuda"
    amp_dtype = (torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16) if use_amp else torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    model.to(device)
    model.train()

    # ---- 数据 ----
    ds = QADataset(args.data, tok, args.seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
                    num_workers=0, collate_fn=lambda b: collate_pad(b, tok.pad_id))
    print(f"问答样本：{len(ds)} 条，设备：{device}")

    decay, nodecay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() >= 2 else nodecay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    def save(path, step):
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                    "config": cfg.to_dict(), "tokenizer_chars": tok.chars,
                    "stage": "sft"}, path)

    print("-" * 66)
    step = 0
    accum = 0
    t0 = time.time()
    smooth = None
    data_iter = iter(dl)

    while step < args.max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            _, loss, _ = model(x, targets=y)
        scaler.scale(loss / args.grad_accum).backward()
        accum += 1

        if accum % args.grad_accum == 0:
            lr = args.lr * (0.5 * (1 + math.cos(math.pi * step / max(1, args.max_steps))
                            if step > args.warmup else (step + 1) / max(1, args.warmup)))
            for g in opt.param_groups:
                g["lr"] = max(lr, args.lr * 0.1)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1

            smooth = loss.item() if smooth is None else 0.9 * smooth + 0.1 * loss.item()
            if step % args.log_every == 0 or step == 1:
                print(f"step {step:5d}/{args.max_steps} | loss {smooth:6.3f} | "
                      f"{time.time() - t0:5.1f}s")
            if args.save_every and step % args.save_every == 0:
                save(os.path.join(args.out_dir, "ckpt_last.pt"), step)

    save(os.path.join(args.out_dir, "ckpt_final.pt"), step)
    save(os.path.join(args.out_dir, "ckpt_last.pt"), step)
    print("-" * 66)
    print(f"微调结束，权重在：{os.path.abspath(args.out_dir)}")
    print("下一步：py -3.13 chat.py --ckpt %s" % os.path.join(args.out_dir, "ckpt_final.pt"))


if __name__ == "__main__":
    main()
