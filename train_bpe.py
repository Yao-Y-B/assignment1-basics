import regex as re
from collections import defaultdict,Counter
from multiprocessing import Pool
import time
import sys
import json
from typing import Iterable, Iterator

from cs336_basics.pretokenization_example import find_chunk_boundaries

'''
def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab dict[int, token_bytes]:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges   list[tuple[bytes, bytes]]:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
'''

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def merge(counts: dict[tuple[int, int], int], indices: list[int], pair: tuple[int, int], new_index: int, cnt: int):
    # counts: pair(int,int)频次
    # indices: 当前单词对应的int序列
    # pair: 本轮要合并的(int,int)
    # new_index: 新token在vocab中对应的idx
    # cnt: 当前单词的词频
    new_indices = [] #合并下单词里面的
    i,flag = 0,1 #flag=1代表i前一个是unmerged
    # (w,i,d,e,s,t)
    # (w,i,d,es,t)
    while i < len(indices):
        if i+1 < len(indices) and indices[i] == pair[0] and indices[i+1] == pair[1]:
            new_indices.append(new_index)

            if i > 0:
                # 减词频
                counts[(indices[i-1], indices[i])] -= cnt
                if flag:
                    counts[(indices[i-1], new_index)] += cnt
                else:
                    # （h,e,h,e）---> (he,he)
                    counts[(new_index, new_index)] += cnt

            flag = 0
            i = i+2

        else:
            new_indices.append(indices[i])
            if i > 0:
                if not flag:
                    counts[(indices[i - 1], indices[i])] -= cnt
                    counts[(new_index, indices[i])] += cnt

            flag = 1
            i += 1

    return new_indices


def word_split(word_cnt: dict[str, int], merge_times: int):

    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    word_idx: dict[str, list] = {}

    counts = defaultdict(int)
    for word, cnt in word_cnt.items():
    #     创建每个词的indices之后用
        indices = list(map(int, word.encode("utf-8")))
        # [104, 101, 108, 108, 111]
        word_idx[word] = indices
        # {
    #     "hello": [104, 101, 108, 108, 111],
    #     "help":  [104, 101, 108, 112],
    # }
        # 统计pair
        for a, b in zip(indices, indices[1:]):
            # 这个字节对（int, int）直接加词频
            counts[(a, b)] += cnt

    # 第一次BPE merge
    for _ in range(merge_times):
        if not counts:
            break

        # 找到词频最高的字节对
        max_value = max(counts.values())
        # idxs: (int, int)   val: int,次数
        # max_keys: (索引，索引)，（字节，字节）
        max_keys = [(idxs, (vocab[idxs[0]],vocab[idxs[1]])) for idxs,val in counts.items() if val == max_value]
        # x[1]:字节对， 只看字节排序，取最高
        int_pair,byte_pair = sorted(max_keys, key = lambda x : x[1], reverse=True)[0]
        # 最高对，频数

        merges.append(byte_pair)
        vocab[len(vocab)] = byte_pair[0] + byte_pair[1]

        counts[int_pair] = 0
        for word, idx in word_idx.items():
            word_idx[word] = merge(counts, idx, int_pair, len(vocab) - 1, word_cnt[word])

    return vocab, merges

def process_chunk(chunk):
    return Counter(re.findall(PAT, chunk))

def pre_tokenization(text: str, special_tokens: list[str] | None = None) -> Counter:
    num_processes = 4
    chunks = re.split("|".join(re.escape(sp_tok) for sp_tok in special_tokens), text)

    if len(chunks) == 1:
        # 只有一个分块组
        origin_text = chunks[0]
        proc_text = re.findall(PAT, origin_text)
        chunk_size = len(proc_text) // num_processes
        bounderies = [i * chunk_size for i in range(num_processes + 1)]
        bounderies[-1] = len(proc_text)
        chunks = [proc_text[i : j] for i, j in zip(bounderies[:-1], bounderies[1:])]
        with Pool(processes=num_processes) as pool:
            counters = pool.map(Counter, chunks)
    else:
    #     多个文本组
        if len(chunks) < num_processes:
            num_processes = len(chunks)
        with Pool(processes=num_processes) as pool:
            counters = pool.map(process_chunk, chunks)

    word_cnt = Counter()
    for counter in counters:
        word_cnt.update(counter)

    # word_cnt : {hello: 1, world: 2}

    return word_cnt

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str] | None = None,):
    # 合并的次数：
    merge_times = vocab_size - len(special_tokens) - 256

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    word_cnt = pre_tokenization(text, special_tokens) #统计词频

    vocab, merges = word_split(word_cnt, merge_times)

    # 添加特殊标记
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    return vocab, merges


class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges

        # 同一个special token的合并
        self.special_tokens = special_tokens

        vocab_val = vocab.values()
        if special_tokens:
            for sp_tok in special_tokens:
                sp_byte = sp_tok.encode('utf-8')
                if sp_byte not in vocab_val:
                    vocab[len(vocab)] = sp_byte

        self.byte_encoder = {v: k for k, v in vocab.items()}
        # self.merge_dict = {pair: idx for idx, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        # json格式储存字典：vocab
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        vocab = {v: bytes(k.encode('utf-8')) for k, v in vocab_data.items()}

        # txt格式储存列表：merges
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ')
                if len(parts) == 2:
                    byte1 = parts[0].encode('utf-8')
                    byte2 = parts[1].encode('utf-8')
                    merges.append((byte1, byte2))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # pre tokenization
        # tokens_list：原始text粗略拆出来的预分词单元（可能有重复）
        if self.special_tokens:
            chunks = re.split("(" + "|".join(
                re.escape(sp_tok) for sp_tok in sorted(self.special_tokens, key=len, reverse=True)) + ")", text)
            chunks = [chunk for chunk in chunks if chunk != ' ']
        else:
            chunks = [text]

        tokens_list, no_special_tokens = [], []
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                tokens_list.append(chunk)
            else:
                tokens_list += re.findall(PAT, chunk)
                no_special_tokens += re.findall(PAT, chunk)

        # 哈希表的形式逐个处理预分词单元
        initial_dict = {word: [bytes([integer]) for integer in word.encode('utf-8')] for word in set(no_special_tokens)}

        # 遍历merges
        for iter in self.merges:
            for word, byte_list in initial_dict.items():
                pos = 0
                new_bl = []
                while pos < len(byte_list):
                    if pos + 1 < len(byte_list) and (byte_list[pos], byte_list[pos + 1]) == iter:
                        new_bl.append(iter[0] + iter[1])
                        pos += 2
                    else:
                        new_bl.append(byte_list[pos])
                        pos += 1
                initial_dict[word] = new_bl

        # 根据哈希表，将bytes映射会原始text
        result = []
        for word in tokens_list:
            bl = initial_dict.get(word, [word.encode('utf-8')])
            result += [self.byte_encoder[single_byte] for single_byte in bl]
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_chunk in iterable:
            tokens = self.encode(text_chunk)
            for token in tokens:
                yield token

    def decode(self, ids: list[int]) -> str:
        # 注意，这个地方给定的vocab和训练的vocab构建方式不一致：训练的vocab前256个一定是单字节，中间是合并后的字节，最后是special tokens
        bytes_out = bytes([])
        for id in ids:
            bytes_out += self.vocab[id]
        return bytes_out.decode("utf-8", errors='replace')

# class BPETokenizer:
#     def __init__(self, vocab : dict[int, bytes], merges : list[tuple[bytes,bytes]], special_tokens : None = None | list[str]):
#         if not isinstance(vocab, dict):
#             print("vocab must be a dict")
#         if len(vocab) == 0:
#             print("vocab is empty")
#         if not isinstance(merges, list):
#             print("merges must be a list")
#
#         self.vocab = vocab
#         self.merges = merges
#         self.special_tokens = special_tokens or []
#         # 处理special tokens: 不在vocab中就加到后面
#         vocab_val = vocab.values()
#         if special_tokens:
#             for tokens in special_tokens:
#                 sp_byte = tokens.encode("utf-8")
#                 if sp_byte not in vocab_val:
#                     vocab[len(vocab)] = sp_byte
#
#         self.vocab_rev = {v:k for k,v in vocab.items()}
#
#
#     @classmethod
#     def from_files(cls, vocab_filepath : str, merges_filepath : str, special_tokens : list[str]|None = None):
#
#         with open(vocab_filepath, "r", encoding="utf-8") as f:
#             raw_vocab = json.load(f)
#         vocab = {int(k) : bytes(v.encode("utf-8")) for k,v in raw_vocab.items()}
#
#         merges : list[tuple[bytes, bytes]] = []
#         with open(merges_filepath, "r", encoding="utf-8") as f:
#             raw_merges = json.load(f)
#         merges = [(a.encode("utf-8"), b.encode("utf-8"))for a, b in list(raw_merges)]
#
#         return cls(vocab,merges,special_tokens)
#
#
#
#
#     def encode(self, text: str)-> list[int]:
#         result: list[int] = []
#
#         # 1. 先保护 special token（如果有）
#         if self.special_tokens:
#             # 生成 special token 的匹配 pattern
#             sp_pat = "(" + "|".join(map(re.escape, self.special_tokens)) + ")"
#             segments = re.split(sp_pat, text)
#         else:
#             segments = [text]
#
#         # 2. 处理每个 segment
#         for seg in segments:
#             if not seg:
#                 continue
#
#             # 如果是 special token，直接加入 result
#             if seg in self.special_tokens:
#                 result.append(self.vocab_rev[seg.encode("utf-8")])
#                 continue
#
#             # 3. 普通文本预切分
#             pre_tokens = re.findall(PAT, seg)
#             for token in pre_tokens:
#                 # 转成初始 byte token list
#                 indices = [bytes([b]) for b in token.encode("utf-8")]
#
#                 # 4. 反复执行 merge
#                 while True:
#                     pairs = [(a, b) for a, b in zip(indices, indices[1:])]
#                     mergeable = next((p for p in pairs if p in self.merges), None)
#                     if not mergeable:
#                         break  # 不能再 merge，退出
#
#                     # 合并
#                     new_token = mergeable[0] + mergeable[1]
#                     new_id = self.vocab_rev[new_token]
#
#                     # 生成新的 indices
#                     new_indices = []
#                     i = 0
#                     while i < len(indices):
#                         if i < len(indices) - 1 and (indices[i], indices[i + 1]) == mergeable:
#                             new_indices.append(self.vocab[new_id])
#                             i += 2
#                         else:
#                             new_indices.append(indices[i])
#                             i += 1
#
#                     indices = new_indices
#
#                 # 5. 将最终 token ID 加入 result（展平）
#                 result.extend(self.vocab_rev[b] for b in indices)
#
#         return result
#
#     def encode_iterable(self,iterable:Iterable[str])->Iterator[int]:
#         for line in iterable:  # 逐行读取，不加载整个文件
#             ids = self.encode(line)
#             for i in ids:
#                 yield i  # 这是 generator
#
#     def decode(self, ids:list[int])-> str:
#         parts = []
#         for i in ids:
#             token_bytes = self.vocab.get(i)
#             if token_bytes is None:
#                 print("token_bytes not found")
#             else:
#                 try:
#                     parts.append(token_bytes.decode("utf-8"))
#                 except:
#                     parts.append(token_bytes.decode("utf-8", errors="replace"))
#         return "".join(parts)

