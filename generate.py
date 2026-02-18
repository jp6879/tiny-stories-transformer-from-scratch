import argparse
import torch
from stories_gpt import GPT, BPETokenizer


def main():
    p = argparse.ArgumentParser(description="Generate text from a trained mini GPT")
    p.add_argument("--prompt", type=str, required=True, help="starting text")
    p.add_argument(
        "--length", type=int, default=200, help="tokens to generate (default: 200)"
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="sampling temperature (default: 0.8)",
    )
    p.add_argument(
        "--top-k", type=int, default=50, help="top-k filtering (default: 50)"
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/gpt_model.pt",
        help="model checkpoint path",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default="checkpoints/tokenizer",
        help="tokenizer path prefix",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = GPT(
        vocab_size=cfg["vocab_size"],
        embedding_dim=cfg["embedding_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        max_len=cfg["max_len"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer)

    ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(
        ids, max_new_tokens=args.length, temperature=args.temperature, top_k=args.top_k
    )
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
