import json
import os
import time
import traceback

from tests.adapters import run_train_bpe

def main():
    # --------- 配置区（一眼可读）---------
    input_path = "./data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 1000
    special_tokens = ["<|endoftext|>"]

    print("==== BPE Training Configuration ====")
    print(f"Corpus        : {input_path}")
    print(f"Vocab size    : {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    print("=" * 36)

    start_time = time.time()

    try:
        print("[1/3] Starting BPE training...")
        vocab, merges = run_train_bpe(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        )
        print("[2/3] BPE training finished.")

    except Exception as e:
        print("\n[ERROR] BPE training failed.")
        print(f"Elapsed time: {time.time() - start_time:.2f}s")
        print("Exception:")
        traceback.print_exc()
        return

    end_time = time.time()

    # --------- 结果摘要 ---------
    print("\n==== Training Summary ====")
    print(f"Total training time : {end_time - start_time:.2f}s")
    print(f"Final vocab size    : {len(vocab)}")
    print(f"Total merges        : {len(merges)}")

    vocab_json = {
        str(token_id): token_bytes.decode("latin-1")
        for token_id, token_bytes in vocab.items()
    }
    merges_json = [
        [
            a.decode("latin-1"),
            b.decode("latin-1")
        ]
        for (a, b) in merges
    ]

    os.makedirs("./vocab", exist_ok=True)
    with open("./vocab/ts_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)
    with open("./vocab/ts_merges.json", "w") as f:
        json.dump(merges_json, f, ensure_ascii=False, indent=2)

    # sanity checks
    print("\n==== Sanity Checks ====")
    for tok in special_tokens:
        tok_bytes = tok.encode("utf-8")
        if tok_bytes in vocab.values():
            print(f"[OK] Special token in vocab: {tok}")
        else:
            print(f"[WARN] Special token NOT found in vocab: {tok}")

    print("\nDone.")


if __name__ == "__main__":
    main()