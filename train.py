# -*- coding: utf-8 -*-
"""
train.py —— 预训练（在纯文本上学"下一个字是什么"）

用法：
    # 小模型，笔记本 CPU 也能跑
    py -3.13 train.py --config configs/tiny.json --data data/corpus.txt --out_dir out_tiny --max_steps 500

    # 1.5B，需要 20GB 以上显存的显卡
    py -3.13 train.py --config configs/1.5b.json --data data/corpus.txt --out_dir out_1.5b --max_steps 20000 --batch_size 1 --grad_accum 16

断点续训：加 --resume out_dir/ckpt_last.pt
"""
import argparse
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from data import LMDataset, read_corpus
from model import GPT, GPTConfig
from tokenizer import CharTokenizer


def pick_device(name, n_params=None, force_cpu=False):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # 没显卡时拦一下：1.5B 在 CPU 上训练一辈子也跑不完
    if n_params and n_params > 100_000_000 and not force_cpu:
        raise SystemExit(
            "检测到你在用 CPU 训练一个 %d 亿参数的模型，这一步基本跑不完（一步要好几分钟）。\n"
            "选择：1) 换 configs/tiny.json 先跑通流程；2) 用带显卡的机器；\n"
            "      3) 确实要硬跑，加参数 --force_cpu" % round(n_params / 1e8)
        )
    return torch.device("cpu")


def get_lr(step, max_steps, lr, min_lr, warmup):
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    if step >= max_steps:
        return min_lr
    p = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * p))


def resolve_vocab(cfg, tok):
    """配置里 vocab_size 为 null 时跟随分词器；否则取两者较大的，避免 id 越界"""
    need = tok.vocab_size
    if cfg.vocab_size is None:
        cfg.vocab_size = (need + 127) // 128 * 128      # 补到 128 的整数倍
    if cfg.vocab_size < need:
        print(f"[提示] 配置词表 {cfg.vocab_size} 小于分词器 {need}，已自动放大")
        cfg.vocab_size = (need + 127) // 128 * 128
    if cfg.vocab_size > need:
        print(f"[提示] 词表预留 {cfg.vocab_size} 行，语料只用到 {need} 个 token，"
              f"空着的行不参与梯度（想让 1.5B 满配就上更大的语料）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tiny.json")
    ap.add_argument("--data", default="data/corpus.txt")
    ap.add_argument("--out_dir", default="out_tiny")
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--min_lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_iters", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--force_cpu", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 配置 ----
    with open(args.config, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = GPTConfig.from_dict(raw)
    if args.seq_len:
        cfg.max_seq_len = args.seq_len

    # ---- 分词器 ----
    if not os.path.exists(args.tokenizer):
        print(f"没有找到 {args.tokenizer}，正在从语料自动构建……")
        tok = CharTokenizer.train_from_files([args.data])
        tok.save(args.tokenizer)
    tok = CharTokenizer.load(args.tokenizer)
    resolve_vocab(cfg, tok)
    tok.save(os.path.join(args.out_dir, "tokenizer.json"))

    # ---- 数据 ----
    text = read_corpus(args.data)
    ids = [tok.bos_id] + tok.encode(text) + [tok.eos_id]
    print(f"语料：{len(text)} 字符 -> {len(ids)} token")
    if len(ids) < cfg.max_seq_len * 4:
        print(f"[警告] 语料太少（不足 {cfg.max_seq_len * 4} token），模型只能背下这几句话，"
              f"想要像样的回答请换成几百 MB 甚至 GB 级的文本")

    n_val = max(cfg.max_seq_len + 1, int(len(ids) * 0.1))
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]
    if len(train_ids) <= cfg.max_seq_len:
        train_ids, val_ids = ids, ids
    ds = LMDataset(train_ids, cfg.max_seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_ds = LMDataset(val_ids, cfg.max_seq_len)

    # ---- 模型 ----
    model = GPT(cfg)
    n_params = model.num_params()
    print(f"模型参数量：{n_params:,}（{n_params / 1e9:.3f} B）")

    device = pick_device(args.device, n_params, args.force_cpu)

    if args.dtype == "auto":
        if device.type == "cuda":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            amp_dtype = torch.float32
    else:
        amp_dtype = getattr(torch, args.dtype)
    use_amp = amp_dtype != torch.float32 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    model.to(device)
    model.train()
    print(f"设备：{device}  精度：{'auto->' + str(amp_dtype).replace('torch.', '') if use_amp else 'float32'}")

    # ---- 优化器：大矩阵衰减，小向量（norm/bias）不衰减 ----
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() >= 2 else nodecay).append(p)
    base_lr = args.lr or float(raw.get("lr", 3e-4))
    min_lr = args.min_lr if args.min_lr is not None else base_lr * 0.1
    warmup = args.warmup if args.warmup is not None else max(10, args.max_steps // 100)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=base_lr, betas=(0.9, 0.95), eps=1e-8,
    )

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_step = ck["step"]
        print(f"从 {args.resume} 续训，已完成 {start_step} 步")

    def save(path, step, best_val=None):
        torch.save({
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            "config": cfg.to_dict(),
            "tokenizer_chars": tok.chars,
            "best_val": best_val,
        }, path)

    def evaluate():
        model.eval()
        losses = []
        with torch.no_grad():
            for i in range(min(args.eval_iters, len(val_ds))):
                x, y = val_ds[i]
                x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    _, loss, _ = model(x, targets=y)
                losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    # ---- 训练循环 ----
    print("-" * 66)
    step = start_step
    accum = 0
    t0 = time.time()
    tokens_done = 0
    smooth = None
    best_val = None
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
        loss_scaled = loss / args.grad_accum
        scaler.scale(loss_scaled).backward()
        accum += 1
        tokens_done += x.numel()

        if accum % args.grad_accum == 0:
            lr = get_lr(step, args.max_steps, base_lr, min_lr, warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1

            loss_val = loss.item()
            smooth = loss_val if smooth is None else 0.9 * smooth + 0.1 * loss_val

            if step % args.log_every == 0 or step == 1:
                dt = time.time() - t0
                print(f"step {step:6d}/{args.max_steps} | loss {smooth:6.3f} | lr {lr:.2e} | "
                      f"{tokens_done / max(dt, 1e-6):7.0f} tok/s | {dt:6.1f}s")

            if args.eval_every and step % args.eval_every == 0:
                val = evaluate()
                flag = ""
                if best_val is None or val < best_val:
                    best_val = val
                    save(os.path.join(args.out_dir, "ckpt_best.pt"), step, best_val)
                    flag = "  <- 新最优，已保存 ckpt_best.pt"
                print(f"  [验证] val_loss {val:.3f}{flag}")

            if args.save_every and step % args.save_every == 0:
                save(os.path.join(args.out_dir, "ckpt_last.pt"), step, best_val)

    save(os.path.join(args.out_dir, "ckpt_last.pt"), step, best_val)
    save(os.path.join(args.out_dir, "ckpt_final.pt"), step, best_val)
    print("-" * 66)
    print(f"训练结束：{step} 步，总耗时 {time.time() - t0:.1f}s")
    print(f"权重已保存到：{os.path.abspath(args.out_dir)}")
    print("下一步：py -3.13 sft.py --ckpt %s --data data/qa.jsonl" % os.path.join(args.out_dir, "ckpt_final.pt"))


if __name__ == "__main__":
    main()
