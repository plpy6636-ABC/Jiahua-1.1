# -*- coding: utf-8 -*-
"""
model.py —— 从零手写的大模型骨架（LLaMA 式 decoder-only Transformer）

只依赖 PyTorch，不用 transformers，每一行都可以看懂。

前向流程：
    词向量表 -> 24 层 [ RMSNorm -> RoPE 多头注意力 -> 残差 -> RMSNorm -> SwiGLU 前馈 -> 残差 ]
             -> 最后 RMSNorm -> LM Head -> 每个位置对 4 万词表打分

查看参数量：
    py -3.13 model.py --config configs/1.5b.json
    py -3.13 model.py --config configs/tiny.json
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. 归一化：RMSNorm（比 LayerNorm 少一半计算，LLaMA 用的就是它）
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dt = x.dtype
        x = x.float()  # 用 fp32 算平方均值，防止 bf16 溢出
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


# ---------------------------------------------------------------------------
# 2. 位置编码：RoPE（旋转位置编码，把"第几个字"转成角度旋进向量里）
# ---------------------------------------------------------------------------
def precompute_rope(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(max_seq_len).float()
    freqs = torch.outer(pos, inv_freq)          # [T, head_dim/2]
    return torch.cos(freqs), torch.sin(freqs)   # 两个 [T, head_dim/2] 的表


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_apply(x, cos, sin):
    """x: [B, H, T, head_dim]；cos/sin: [T, head_dim/2]"""
    cos = torch.cat([cos, cos], dim=-1).to(x.dtype)[None, None]
    sin = torch.cat([sin, sin], dim=-1).to(x.dtype)[None, None]
    return x * cos + rotate_half(x) * sin


# ---------------------------------------------------------------------------
# 3. 注意力：支持 KV Cache（生成时不用每次重算整段历史）
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head or cfg.n_head
        assert cfg.d_model % cfg.n_head == 0, "d_model 必须能被 n_head 整除"
        assert self.n_head % self.n_kv_head == 0, "n_head 必须是 n_kv_head 的整数倍"
        self.head_dim = cfg.d_model // cfg.n_head
        self.group = self.n_head // self.n_kv_head  # GQA：多个 Q 头共享一组 K/V

        self.q_proj = nn.Linear(cfg.d_model, self.n_head * self.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_head * self.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_head * self.head_dim, bias=cfg.bias)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, cfg.d_model, bias=cfg.bias)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        B, T, _ = x.shape
        past_len = 0 if past_kv is None else past_kv[0].size(2)

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # 新词的位置是 past_len .. past_len+T-1（q 和 k 都是这批新 token，
        # 缓存的是已经算过的旧 k，不要在这里拼全长，否则张量会被广播错）
        q = rope_apply(q, cos[past_len:past_len + T], sin[past_len:past_len + T])
        k = rope_apply(k, cos[past_len:past_len + T], sin[past_len:past_len + T])

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        if self.group > 1:
            k = k.repeat_interleave(self.group, dim=1)
            v = v.repeat_interleave(self.group, dim=1)

        Tq, Tk = q.size(2), k.size(2)
        mask = None
        if Tq > 1:  # 训练时一次看很多词，需要下三角掩码；生成时 Tq=1 不用
            mask = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device).tril(diagonal=Tk - Tq)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(B, Tq, -1)
        return self.o_proj(out), ((k, v) if use_cache else None)


# ---------------------------------------------------------------------------
# 4. 前馈网络：SwiGLU（LLaMA 用的门控结构，效果比普通 MLP 好）
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.ffn_hidden
        self.gate = nn.Linear(cfg.d_model, h, bias=cfg.bias)
        self.up = nn.Linear(cfg.d_model, h, bias=cfg.bias)
        self.down = nn.Linear(h, cfg.d_model, bias=cfg.bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        h, kv = self.attn(self.attn_norm(x), cos, sin, past_kv, use_cache)
        x = x + h
        x = x + self.mlp(self.mlp_norm(x))
        return x, kv


# ---------------------------------------------------------------------------
# 5. 配置
# ---------------------------------------------------------------------------
class GPTConfig:
    def __init__(
        self,
        vocab_size=None,
        n_layer=24,
        n_head=12,
        n_kv_head=None,
        d_model=1536,
        ffn_hidden=None,
        max_seq_len=1024,
        dropout=0.0,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        norm_eps=1e-6,
        bias=False,
        init_std=0.02,
    ):
        self.vocab_size = vocab_size            # None = 跟随分词器实际词表大小
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head or n_head
        self.d_model = d_model
        self.ffn_hidden = ffn_hidden or int(round((8 / 3) * d_model / 128) * 128)
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.norm_eps = norm_eps
        self.bias = bias
        self.init_std = init_std

    @classmethod
    def from_dict(cls, d):
        if d is None:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls().__dict__})

    def to_dict(self):
        return dict(self.__dict__)

    def __repr__(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ---------------------------------------------------------------------------
# 6. 完整模型
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight  # 输入输出共用一张词向量表，省参数

        cos, sin = precompute_rope(cfg.d_model // cfg.n_head, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # 残差分支的输出层缩小初始化，deep network 训练更稳
        std = cfg.init_std / math.sqrt(2 * cfg.n_layer)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=std)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if not self.cfg.tie_word_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def forward(self, idx, targets=None, past_kv=None, use_cache=False):
        """
        idx       : [B, T] 的 token id
        targets   : [B, T]，-100 表示这个位置不算损失
        past_kv   : list，每层一个 (k, v)，生成时复用
        返回       : (logits, loss, new_past_kv)
        """
        B, T = idx.shape
        past_len = 0 if past_kv is None else past_kv[0][0].size(2)
        assert past_len + T <= self.cfg.max_seq_len, (
            f"超长：已 {past_len + T} token，模型上限 {self.cfg.max_seq_len}"
        )

        x = self.emb_drop(self.tok_emb(idx))
        cos = self.rope_cos[:past_len + T]
        sin = self.rope_sin[:past_len + T]

        new_kv = [] if use_cache else None
        for i, blk in enumerate(self.blocks):
            x, kv = blk(x, cos, sin, None if past_kv is None else past_kv[i], use_cache)
            if use_cache:
                new_kv.append(kv)

        x = self.norm_f(x)
        logits = self.lm_head(x) if not self.cfg.tie_word_embeddings else F.linear(x, self.tok_emb.weight)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss, new_kv

    # ------------------------------------------------------------------
    # 生成：温度 / top-k / top-p / 重复惩罚
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=64, temperature=0.8, top_k=50, top_p=0.9,
                 repetition_penalty=1.0, banned_ids=None):
        self.eval()
        past = None
        out = idx
        for _ in range(max_new_tokens):
            cur = out if past is None else out[:, -1:]
            logits, _, past = self(cur, past_kv=past, use_cache=True)
            logits = logits[:, -1, :].float()

            if repetition_penalty != 1.0:
                for b in range(out.size(0)):
                    uniq = torch.unique(out[b])
                    logits[b, uniq] /= repetition_penalty

            if banned_ids:
                logits[:, list(banned_ids)] = -float("inf")

            if temperature <= 0:  # 贪心
                nxt = logits.argmax(dim=-1, keepdim=True)
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
                    rm = cum - sp > top_p
                    sp[rm] = 0.0
                    probs = torch.zeros_like(probs).scatter(-1, si, sp)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                nxt = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, nxt], dim=1)
        return out


# ---------------------------------------------------------------------------
# 7. 命令行：打印参数量 / 结构
# ---------------------------------------------------------------------------
def human(n):
    if n >= 1e9:
        return f"{n / 1e9:.3f} B"
    if n >= 1e6:
        return f"{n / 1e6:.1f} M"
    return f"{n / 1e3:.1f} K"


def main():
    ap = argparse.ArgumentParser(description="查看模型结构与参数量（不占内存，用 meta 设备构建）")
    ap.add_argument("--config", default="configs/1.5b.json")
    ap.add_argument("--vocab_size", type=int, default=None, help="覆盖配置里的词表大小")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = GPTConfig.from_dict(json.load(f))
    if args.vocab_size:
        cfg.vocab_size = args.vocab_size
    if cfg.vocab_size is None:
        p = os.path.join(os.path.dirname(os.path.abspath(args.config)), "..", "tokenizer.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                cfg.vocab_size = len(json.load(f)["chars"]) + 4
        else:
            cfg.vocab_size = 4096

    with torch.device("meta"):
        model = GPT(cfg)

    total = model.num_params()
    blocks = sum(p.numel() for n, p in model.named_parameters() if n.startswith("blocks"))
    emb = model.tok_emb.weight.numel()
    head = 0 if cfg.tie_word_embeddings else model.lm_head.weight.numel()

    print("=" * 66)
    print(f"配置文件        : {args.config}")
    print(f"层数 n_layer    : {cfg.n_layer}")
    print(f"隐藏维度 d_model: {cfg.d_model}    头数 n_head: {cfg.n_head}  每头维度: {cfg.d_model // cfg.n_head}")
    print(f"前馈维度 ffn    : {cfg.ffn_hidden}  (SwiGLU)")
    print(f"词表 vocab      : {cfg.vocab_size}")
    print(f"上下文 max_seq  : {cfg.max_seq_len}")
    print("-" * 66)
    print(f"Transformer 主体: {human(blocks)}")
    print(f"词向量表        : {human(emb)}" + ("（与输出层共享）" if cfg.tie_word_embeddings else ""))
    if head:
        print(f"输出层          : {human(head)}")
    print("-" * 66)
    print(f"总参数量        : {total:,}  ( {human(total)} )")
    print(f"参数显存(fp16)  : 约 {total * 2 / 1024**3:.2f} GB")
    print(f"训练显存(AdamW) : 约 {total * 2 / 1024**3 + total * 4 * 3 / 1024**3:.1f} GB 起（还要加激活值）")
    print("=" * 66)


if __name__ == "__main__":
    main()
