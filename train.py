import os
import time
import json
import torch
from torch.utils.data import DataLoader

from stories_gpt import GPT, BPETokenizer
from dataset import download_tinystories, TextDataset

with open("configurations/config.json", "r") as f:
    cfg = json.load(f)

VOCAB_SIZE = cfg["vocab_size"]
EMBED_DIM = cfg["embedding_dim"]
NUM_HEADS = cfg["num_heads"]
NUM_LAYERS = cfg["num_layers"]
SEQ_LEN = cfg["seq_len"]
BATCH_SIZE = cfg["batch_size"]
EPOCHS = cfg["epochs"]
LR = cfg["lr"]
DROPOUT = cfg["dropout"]
TRAIN_STORIES = cfg["num_train_stories"]
VAL_STORIES = cfg["num_val_stories"]
CKPT_DIR = cfg["checkpoint_dir"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 1. data
    train_text, val_text = download_tinystories(
        num_train=TRAIN_STORIES,
        num_val=VAL_STORIES,
    )

    # 2. tokenizer (train once, then load from cache)
    tokenizer = BPETokenizer()
    tok_path = os.path.join(CKPT_DIR, "tokenizer")
    if os.path.exists(f"{tok_path}.merges"):
        tokenizer.load(tok_path)
    else:
        print("Training BPE tokenizer...")
        tokenizer.train(train_text, vocab_size=VOCAB_SIZE)
    print(f"vocab: {len(tokenizer.vocab)} tokens ({len(tokenizer.merges)} merges)")

    # 3. tokenize
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    print(f"tokens — train: {len(train_ids):,}  val: {len(val_ids):,}")

    # 4. dataloaders
    train_loader = DataLoader(
        TextDataset(train_ids, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(TextDataset(val_ids, SEQ_LEN), batch_size=BATCH_SIZE)
    print(f"batches — train: {len(train_loader)}  val: {len(val_loader)}")

    # 5. model
    model = GPT(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_len=SEQ_LEN,
        dropout=DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")

    # 6. optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    total_steps = EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    # 7. train
    best_val_loss = float("inf")
    print(f"\ntraining for {EPOCHS} epochs ({total_steps} steps)\n")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        t0 = time.time()

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            loss = loss_fn(model(x).view(-1, VOCAB_SIZE), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            if (step + 1) % 50 == 0:
                avg = running_loss / (step + 1)
                lr = scheduler.get_last_lr()[0]
                print(
                    f"[{epoch+1}/{EPOCHS}] step {step+1}/{len(train_loader)}\tloss {avg:.4f}\tlr {lr:.2e}"
                )

        train_loss = running_loss / max(len(train_loader), 1)
        dt = time.time() - t0

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += loss_fn(model(x).view(-1, VOCAB_SIZE), y.view(-1)).item()
        val_loss /= max(len(val_loader), 1)

        print(
            f"epoch {epoch+1}/{EPOCHS}\ttrain={train_loss:.4f}\tval={val_loss:.4f}\t{dt:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(CKPT_DIR, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "config": {
                        "vocab_size": VOCAB_SIZE,
                        "embedding_dim": EMBED_DIM,
                        "num_heads": NUM_HEADS,
                        "num_layers": NUM_LAYERS,
                        "max_len": SEQ_LEN,
                        "dropout": DROPOUT,
                    },
                },
                os.path.join(CKPT_DIR, "gpt_model.pt"),
            )
            tokenizer.save(os.path.join(CKPT_DIR, "tokenizer"))
            print(f"  saved (val_loss={val_loss:.4f})")

    # 8. sample
    print("\n" + "=" * 50)
    print("sample generation:")
    print("=" * 50)
    model.eval()
    prompt = "Once upon a time"
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=200, temperature=0.8, top_k=50)
    print(f"\n> {prompt}")
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
