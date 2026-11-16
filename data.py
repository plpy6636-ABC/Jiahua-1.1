# -*- coding: utf-8 -*-
"""
data.py —— 训练数据 + 问答模板

两种数据：
1) 预训练（train.py）：data/corpus.txt，一整段纯文本，模型学"下一个字是什么"。
2) 问答微调（sft.py）：data/qa.jsonl，一行一个 {"q": "问题", "a": "回答"}，模型学问答格式。

问答模板（chat.py / sft.py 必须一致）：
    <|用户|>问题<|助手|>回答<|结束|>
"""
import json
import os

import torch
from torch.utils.data import Dataset

USER_TAG = "<|用户|>"
BOT_TAG = "<|助手|>"
END_TAG = "<|结束|>"


def build_prompt(messages):
    """messages: [("user", "问题"), ("assistant", "回答"), ...]，最后一条必须是 user"""
    s = ""
    for role, text in messages:
        if role == "user":
            s += USER_TAG + text
        else:
            s += BOT_TAG + text + END_TAG
    if messages[-1][0] == "user":
        s += BOT_TAG            # 留给模型接着写
    return s


# ---------------------------------------------------------------------------
# 预训练数据：把整段语料切成固定长度的块
# ---------------------------------------------------------------------------
class LMDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len
        step = seq_len
        # 语料太小时窗口数会很少，允许重叠，保证能凑够一个 batch
        n = max(1, (len(tokens) - seq_len - 1) // step + 1)
        if n < 32 and len(tokens) > seq_len + 1:
            step = max(1, (len(tokens) - seq_len - 1) // 32)
        self.starts = list(range(0, max(1, len(tokens) - seq_len - 1), step))
        if len(self.starts) == 0:
            self.starts = [0]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        chunk = self.tokens[s:s + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]                      # 输入右移一位就是标签
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ---------------------------------------------------------------------------
# 问答微调数据：只在"回答"部分算损失
# ---------------------------------------------------------------------------
class QADataset(Dataset):
    def __init__(self, path, tok, seq_len):
        self.tok = tok
        self.seq_len = seq_len
        self.items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                obj = json.loads(line)
                q = str(obj["q"]).strip()
                a = str(obj["a"]).strip()
                prompt = build_prompt([("user", q)])
                answer = a + END_TAG
                p_ids = tok.encode(prompt)
                a_ids = tok.encode(answer) + [tok.eos_id]
                full = p_ids + a_ids
                # 每个位置的"应该预测什么"：问题部分不算，回答部分才算
                lab = [-100] * len(p_ids) + a_ids
                if len(full) > seq_len:      # 太长就砍掉开头的提示，保留问题尾巴和回答
                    cut = len(full) - seq_len
                    full, lab = full[cut:], lab[cut:]
                # 右移一位：第 i 个位置的输出，预测的是第 i+1 个 token
                ids, labels = full[:-1], lab[1:]
                self.items.append((ids, labels))
        if not self.items:
            raise ValueError(f"{path} 里没有读到任何问答数据")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ids, labels = self.items[i]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_pad(batch, pad_id):
    """右填充（因果注意力下右填充是安全的：真词看不到右边的 pad）"""
    maxlen = max(len(x) for x, _ in batch)
    xs, ys = [], []
    for x, y in batch:
        n = maxlen - len(x)
        xs.append(torch.cat([x, torch.full((n,), pad_id, dtype=torch.long)]))
        ys.append(torch.cat([y, torch.full((n,), -100, dtype=torch.long)]))
    return torch.stack(xs), torch.stack(ys)


def read_corpus(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到语料文件：{path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
