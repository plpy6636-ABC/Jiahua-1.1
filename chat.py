# -*- coding: utf-8 -*-
"""
chat.py —— 嘉校1.1 的问答脚本（推理）

用法：
    # 单次提问（1.5B 权重，8G 内存机器加 --quant8 走 int8）
    py -3.13 chat.py --ckpt out_1.5b_sft/ckpt_final.pt --quant8 --question "你好，你是谁？"

    # 连续对话（多轮、记住上下文）
    py -3.13 chat.py --ckpt out_1.5b_sft/ckpt_final.pt --quant8
    输入 退出 结束对话，输入 清空 忘掉之前的内容

说明：
    中文、英文都直接问，模型按你用的语言回答（由训练数据决定，不需要加参数）。
    --quant8 把除了词向量表以外的线性层压成 int8，1.5B 的权重从约 6GB 降到约 1.7GB，
    这是 8G 内存笔记本能跑 1.5B 的关键；代价是速度慢一些，精度几乎不变。
"""
import argparse
import os
import time
import warnings

import torch
import torch.nn.functional as F

# int8 量化在部分 PyTorch 版本会打弃用警告，挡掉，别吓到用户
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")

from data import BOT_TAG, END_TAG, USER_TAG, build_prompt
from model import GPT, GPTConfig
from tokenizer import CharTokenizer


def load_bundle(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig.from_dict(ck["config"])
    tok = CharTokenizer(ck.get("tokenizer_chars") or CharTokenizer.load("tokenizer.json").chars)
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, tok, cfg


def quantize_int8(model, cfg):
    """把线性层动态量化成 int8。权重绑定时先摘掉输出层，避免多存一份 int8 副本。"""
    if cfg.tie_word_embeddings:
        model.lm_head = torch.nn.Identity()   # 前向会走 F.linear(x, tok_emb.weight)
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def weights_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2


def split_params(model, raw_mb):
    """量化前统计：有多少参数在 Linear 里（量化后 1 字节），其余（词向量/norm）保持 fp32"""
    seen, linear = set(), 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            for p in m.parameters(recurse=False):
                if id(p) not in seen:
                    seen.add(id(p))
                    linear += p.numel()
    other_bytes = raw_mb * 1024 ** 2 - linear * 4
    return linear, max(other_bytes, 0)


def _safe_print_len(text):
    """结尾如果正好和 <|结束|> 的开头重合，就先别打印，免得把标记露出来"""
    best = 0
    for k in range(1, min(len(END_TAG), len(text)) + 1):
        if text[-k:] == END_TAG[:k]:
            best = k
    return len(text) - best


@torch.no_grad()
def stream_answer(model, tok, prompt, max_new_tokens=128, temperature=0.8,
                  top_k=40, top_p=0.9, rep_pen=1.15, show=True):
    """逐字生成并打印，返回完整回答文本"""
    device = next(model.parameters()).device
    # 注意：这里不能加 <bos>，必须和 sft.py 里的编码方式完全一致
    ids = tok.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    past = None
    text = ""
    printed = 0
    banned = [tok.pad_id, tok.bos_id, tok.unk_id]
    max_len = model.cfg.max_seq_len
    t0 = time.time()

    for _ in range(max_new_tokens):
        if (0 if past is None else past[0][0].size(2)) + 1 > max_len:
            break
        cur = idx if past is None else idx[:, -1:]
        logits, _, past = model(cur, past_kv=past, use_cache=True)
        logits = logits[:, -1, :].float()

        if rep_pen != 1.0:
            uniq = torch.unique(idx[0])
            logits[0, uniq] = torch.where(logits[0, uniq] > 0,
                                          logits[0, uniq] / rep_pen, logits[0, uniq] * rep_pen)
        if banned:
            logits[:, banned] = -float("inf")

        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k and top_k > 0:
                k = min(top_k, logits.size(-1))
                thresh = torch.topk(logits, k, dim=-1).values[:, -1:]
                logits = torch.where(logits < thresh, torch.full_like(logits, -float("inf")), logits)
            probs = F.softmax(logits, dim=-1)
            if top_p and top_p < 1.0:
                sp, si = torch.sort(probs, dim=-1, descending=True)
                cum = torch.cumsum(sp, dim=-1)
                sp[cum - sp > top_p] = 0.0
                probs = torch.zeros_like(probs).scatter(-1, si, sp)
            probs = probs / probs.sum(-1, keepdim=True)
            nxt = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, nxt], dim=1)
        new_char = tok.decode([nxt.item()])
        if new_char:
            text += new_char
            if show:
                safe = _safe_print_len(text)
                while printed < safe and printed < len(text):
                    print(text[printed], end="", flush=True)
                    printed += 1
        if END_TAG in text or nxt.item() == tok.eos_id:
            break

    visible = text.split(END_TAG)[0].split(USER_TAG)[0]
    if show:
        while printed < len(visible):
            print(visible[printed], end="", flush=True)
            printed += 1
    answer = visible.strip()
    if show:
        print(f"\n[用时 {time.time() - t0:.1f}s，生成 {len(answer)} 字]")
    return answer


def build_history_prompt(history, tok, max_ctx, reserve):
    """历史太长就丢掉最早的对话，保证 prompt 不会超出上下文长度"""
    budget = max_ctx - reserve
    while True:
        prompt = build_prompt(history)
        if len(tok.encode(prompt)) <= budget or len(history) <= 1:
            return prompt
        history = history[2:]   # 一次丢一轮（用户+助手）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out_1.5b_sft/ckpt_final.pt",
                    help="权重文件；试跑版可传 out_tiny_sft/ckpt_final.pt")
    ap.add_argument("--question", nargs="*", default=None, help="一次性提问；不填就进入连续对话")
    ap.add_argument("--quant8", action="store_true",
                    help="int8 动态量化（1.5B 必开，8G 内存才装得下）")
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = 每次都给同一个答案（贪心）")
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--rep_pen", type=float, default=1.15)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit(
            f"找不到权重：{args.ckpt}\n"
            "先在当前目录训练：一键训练1.5B.bat（Windows）或\n"
            "  py -3.13 train.py --config configs/1.5b.json --data data/corpus.txt --out_dir out_1.5b\n"
            "  py -3.13 sft.py   --ckpt out_1.5b/ckpt_last.pt --data data/qa.jsonl --out_dir out_1.5b_sft"
        )

    model, tok, cfg = load_bundle(args.ckpt)

    # ---- 设备：int8 只能 CPU；显存不足 4GB 自动退回 CPU ----
    if args.device != "auto":
        device = torch.device(args.device)
    elif args.quant8:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        if total < 4 * 1024 ** 3:
            print("[提示] 显卡显存不足 4GB，装不下 1.5B，自动改用 CPU（建议同时加 --quant8）")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.quant8 and device.type == "cuda":
        print("[提示] int8 量化只在 CPU 上生效，已强制切到 CPU")
        device = torch.device("cpu")

    raw_mb = weights_mb(model)
    n_params = model.num_params()          # 量化后参数变成打包形式，必须提前算
    linear_numel, other_bytes = split_params(model, raw_mb)
    if args.quant8:
        model = quantize_int8(model, cfg)
        print("[int8] 已量化：线性层 1 字节/参数，词向量表仍是 fp32")

    model.to(device).eval()
    after_mb = (linear_numel + other_bytes) / 1024 ** 2 if args.quant8 else weights_mb(model)
    gen_kw = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                  top_k=args.top_k, top_p=args.top_p, rep_pen=args.rep_pen)

    print("=" * 66)
    print(f"权重：{args.ckpt}")
    print(f"结构：{cfg.n_layer} 层 / {cfg.d_model} 维 / 参数量 {n_params:,}")
    print(f"设备：{device}   量化：{'int8' if args.quant8 else '无'}")
    print(f"权重占用：{after_mb:.0f} MB" + (f"（量化前 {raw_mb:.0f} MB）" if args.quant8 else ""))
    print("=" * 66)

    # ---------- 一次性提问 ----------
    if args.question:
        q = " ".join(args.question)
        print(f"\n用户：{q}\n助手：", end="")
        stream_answer(model, tok, build_prompt([("user", q)]), **gen_kw)
        return

    # ---------- 连续对话 ----------
    print("进入对话（中文英文都可以问；输入 退出 结束，输入 清空 忘记之前的内容）\n")
    history = []
    while True:
        try:
            q = input("用户：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("退出", "exit", "quit"):
            break
        if q == "清空":
            history = []
            print("（已清空上下文）\n")
            continue

        history.append(("user", q))
        prompt = build_history_prompt(history, tok, cfg.max_seq_len, args.max_new_tokens)
        print("助手：", end="")
        answer = stream_answer(model, tok, prompt, **gen_kw)
        history.append(("assistant", answer))
        print()

    print("对话结束。")


if __name__ == "__main__":
    main()
