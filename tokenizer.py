# -*- coding: utf-8 -*-
"""
tokenizer.py —— 字符级分词器（零依赖、纯 Python、中文友好）

原理：把语料里出现过的每一个字符编成一张"字典"，一个字 = 一个 token。
      中文一个字就是一个词，所以字符级对中文天然够用；英文会按字母切，效率低一些。

命令：
    py -3.13 tokenizer.py build  --data data/corpus.txt data/qa.jsonl --out tokenizer.json
    py -3.13 tokenizer.py test   --out tokenizer.json
"""
import argparse
import json
import os

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = [PAD, UNK, BOS, EOS]
# 对话模板用到的符号，语料里不一定出现，但词表里必须有，否则标记会被当成 <unk>
RESERVED_CHARS = list("<|>")


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)                       # 真实字符（不含特殊符号）
        self.itos = list(SPECIALS) + self.chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.unk_id = self.stoi[UNK]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]

    # ------------------------- 建词表 -------------------------
    @classmethod
    def train_from_files(cls, paths, min_count=1):
        counts = {}
        for c in RESERVED_CHARS:
            counts[c] = 1
        for p in paths:
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            if p.endswith(".jsonl"):        # QA 数据：只取 q 和 a 的内容
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    s = str(obj.get("q", "")) + str(obj.get("a", ""))
                    for ch in s:
                        counts[ch] = counts.get(ch, 0) + 1
            else:
                for ch in text:
                    counts[ch] = counts.get(ch, 0) + 1
        # 按出现频次从高到低排序，高频字拿小 id
        chars = [c for c, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if n >= min_count]
        return cls(chars)

    # ------------------------- 存 / 读 -------------------------
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f)["chars"])

    @property
    def vocab_size(self):
        return len(self.itos)

    # ------------------------- 编 / 解 -------------------------
    def encode(self, text, add_bos=False, add_eos=False):
        ids = [self.stoi.get(ch, self.unk_id) for ch in text]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            i = int(i)
            if i < len(SPECIALS):
                continue                     # 特殊符号不显示
            out.append(self.itos[i])
        return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="从语料训练词表")
    b.add_argument("--data", nargs="+", required=True)
    b.add_argument("--out", default="tokenizer.json")
    b.add_argument("--min_count", type=int, default=1)

    t = sub.add_parser("test", help="看词表长什么样")
    t.add_argument("--out", default="tokenizer.json")

    args = ap.parse_args()

    if args.cmd == "build":
        tok = CharTokenizer.train_from_files(args.data, args.min_count)
        tok.save(args.out)
        print(f"词表已保存到 {args.out}：共 {tok.vocab_size} 个 token（4 个特殊符号 + {len(tok.chars)} 个字符）")
    else:
        tok = CharTokenizer.load(args.out)
        s = "你好，我是一个从零训练的小模型。"
        ids = tok.encode(s)
        print("原文 :", s)
        print("id   :", ids)
        print("还原 :", tok.decode(ids))
        print("词表大小:", tok.vocab_size)


if __name__ == "__main__":
    main()
