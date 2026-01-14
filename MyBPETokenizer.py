import regex as re
import json
from typing import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class MyBPETokenizer:
    def __init__(self, vocab : dict[int, bytes], merges : list[tuple[bytes,bytes]], special_tokens : list[str] | None = None ):
        if not isinstance(vocab, dict):
            print("vocab must be a dict")
        if len(vocab) == 0:
            print("vocab is empty")
        if not isinstance(merges, list):
            print("merges must be a list")

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        # 处理special tokens: 不在vocab中就加到后面
        vocab_val = vocab.values()
        if special_tokens:
            for tokens in special_tokens:
                sp_byte = tokens.encode("utf-8")
                if sp_byte not in vocab_val:
                    vocab[len(vocab)] = sp_byte

        self.vocab_rev = {v:k for k,v in vocab.items()}


    @classmethod
    def from_files(cls, vocab_filepath : str, merges_filepath : str, special_tokens : list[str]|None = None):

        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {int(k) : bytes(v.encode("utf-8")) for k,v in raw_vocab.items()}

        with open(merges_filepath, "r", encoding="utf-8") as f:
            raw_merges = json.load(f)
        merges = [(a.encode("utf-8"), b.encode("utf-8"))for a, b in list(raw_merges)]

        return cls(vocab,merges,special_tokens)


    def encode(self, text: str)-> list[int]:

        result: list[int] = []

        # 1. 先保护 special token（如果有）
        if self.special_tokens:
            # 生成 special token 的匹配 pattern
            # sp_pat = "(" + "|".join(map(re.escape, self.special_tokens)) + ")"
            # sorted(self.special_tokens, key=len, reverse=True)：
            # 按照 special token 的长度从大到小排序，这样可以确保长的 token 先被匹配
            segments = re.split("(" + "|".join(re.escape(sp_tok) for sp_tok in sorted(self.special_tokens, key=len, reverse=True))+")", text)
            segments = [seg for seg in segments if seg]
        else:
            segments = [text]

        tokens_list = []
        tokens_without_special = []

        # 2. 处理每个 segment
        for seg in segments:
            if not seg:
                continue

            # 如果是 special token，直接加入 result
            if self.special_tokens and seg in self.special_tokens:
                tokens_list.append(seg)
            else:
                tokens_list += re.findall(PAT, seg)
                tokens_without_special += re.findall(PAT, seg)

        # 3. 普通文本预切分
            # 转成初始 byte token list
        word_dict = {token : [bytes([b]) for b in token.encode("utf-8")]for token in set(tokens_without_special)}
        # 只能通过按merges（list）顺序遍历来解决合并顺序的问题
        for merge_pair in self.merges:
            for word, indices in word_dict.items():
                i = 0
                new_indices = []
                while i < len(indices):
                    if i < len(indices) - 1 and (indices[i], indices[i + 1]) == merge_pair:
                        new_indices.append(merge_pair[0] + merge_pair[1])
                        i += 2
                    else:
                        new_indices.append(indices[i])
                        i += 1

                indices = new_indices
                word_dict[word] = indices

        # # 4. 反复执行 merge
        # for word in word_dict.keys():
        #     indices = word_dict[word]
        #     while True:
        #         pairs = [(a, b) for a, b in zip(indices, indices[1:])]
        #         # mergeable = next((p for p in pairs if p in self.merges), None)
        #         # if not mergeable:
        #         #     break  # 不能再 merge，看下一个词吧
        #         mergeable = min(pairs, key=lambda p: self.merges.get(p, float('inf')), default=None)
        #
        #         # 2. 检查选出的这个最小 rank 的 pair 是否真的在 self.merges 中
        #         # 如果 min 返回的是一个不在 merges 里的 pair（rank 为 inf），说明没法再合并了
        #         if not mergeable or mergeable not in self.merges:
        #             break
        #
        #         # 合并
        #         new_token = mergeable[0] + mergeable[1]
        #         # new_id = self.vocab_rev[new_token]
        #
        #         # 生成新的 indices
        #         new_indices = []
        #         i = 0
        #         while i < len(indices):
        #             if i < len(indices) - 1 and (indices[i], indices[i + 1]) == mergeable:
        #                 new_indices.append(new_token)
        #                 i += 2
        #             else:
        #                 new_indices.append(indices[i])
        #                 i += 1
        #
        #         indices = new_indices
        #         word_dict[word] = indices

        result = []
        for word in tokens_list:
            bl = word_dict.get(word, [word.encode('utf-8')])
            result += [self.vocab_rev[single_byte] for single_byte in bl]
        return result
        # # 5. 将最终 token ID 加入 result（展平）
        # for word in tokens_list:
        #     final_idxs = [ i for i in word_dict[word]]
        #     result += [self.vocab_rev[b] for b in final_idxs]
        # return result

    def encode_iterable(self,iterable:Iterable[str])->Iterator[int]:
        for line in iterable:  # 逐行读取，不加载整个文件
            ids = self.encode(line)
            for i in ids:
                yield i  # 这是 generator

    def decode(self, ids:list[int])-> str:
        output_bytes = bytes([])
        for id in ids:
            output_bytes += self.vocab[id]
        return output_bytes.decode("utf-8",errors="replace")


if __name__ == "__main__":
    vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
    merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
    tokenizer = MyBPETokenizer(vocab,merges)
    print(tokenizer.encode("the cat ate"))