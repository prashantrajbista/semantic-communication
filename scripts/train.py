"""Download the Edinburgh DataShare speech set (if missing), train DeepSC-S, save
checkpoints, and log to Weights & Biases.

W&B key resolution: reads WANDB_API_KEY from a `.env` file in the project root if
present, else from the environment. If neither is set, training continues without
logging in (W&B run goes into disabled/offline mode instead of prompting/failing).

Usage:
    uv run python scripts/train.py --epochs 40 --channel awgn
    uv run python scripts/train.py --epochs 40 --channel rayleigh --subset-size 2000
"""
from __future__ import annotations

import argparse
import os
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepscs.audio import F, L, W
from deepscs.channel import ChannelLayer
from deepscs.data import SpeechDataset, ToyDataset
from deepscs.model import DeepSC_S

ROOT = Path(__file__).resolve().parent.parent
# DataShare's old /bitstream/handle/<id>/<name>.zip URLs now 404/serve an HTML app
# shell instead of the file; the current DSpace 7 API serves bitstreams by UUID
# (found via https://datashare.ed.ac.uk/handle/10283/2791 -> "Download" links).
# 28spk set matches the plan's "~10k train / 824 test" (56spk is a larger variant).
DATASET_BITSTREAMS = {
    "clean_trainset_28spk_wav": "245452b6-6235-44b6-a6f9-e7eb19797769",
    "clean_testset_wav": "dec213d3-bf57-4777-9663-c24bdce92d5e",
}


def load_env_key(name: str, env_path: Path = ROOT / ".env") -> str | None:
    """Tiny .env reader (KEY=VALUE lines) — avoids adding python-dotenv for this."""
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == name:
                return v.strip().strip("'\"")
    return os.environ.get(name)


def _fetch(url: str, zip_path: Path, attempts: int = 3):
    """GET with a browser-like User-Agent (bare urllib UA gets 403'd by some WAFs)
    and a couple of retries with backoff, since DataShare's per-IP rate limiting
    (X-RateLimit-* headers) can 403 transiently — this is more likely on shared/
    datacenter IP ranges (e.g. Colab, cloud VMs) than on a residential connection."""
    import time

    last_err = None
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as out:
                total = int(resp.headers.get("Content-Length", 0))
                read = 0
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
                    read += len(chunk)
                    if total:
                        print(f"\r  {read * 100 / total:5.1f}%", end="", flush=True)
            print()
            return
        except urllib.error.HTTPError as e:
            last_err = e
            print(f"\nattempt {attempt}/{attempts} failed: {e}")
            if attempt < attempts:
                time.sleep(5 * attempt)
    raise RuntimeError(
        f"download failed after {attempts} attempts ({last_err!r}). "
        "If this is a 403 on a cloud/hosted notebook (Colab, etc.), DataShare is "
        "likely rate-limiting or blocking that shared IP range — download the zip "
        "manually in a browser (or via your own Drive), then pass its path with "
        "--train-zip/--test-zip to skip the network fetch."
    )


def download_and_extract(zip_name: str, dest_dir: Path, local_zip: str | None = None):
    """Fetch `<zip_name>.zip` from the Edinburgh DataShare (or use a pre-downloaded
    `local_zip`) and extract into data/. ponytail: assumes the zip extracts directly
    into `<zip_name>/*.wav`; several GB, so this is skipped entirely if dest_dir
    already has .wav files."""
    if dest_dir.exists() and any(dest_dir.glob("*.wav")):
        print(f"{dest_dir} already populated, skipping download.")
        return
    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    if local_zip:
        zip_path = Path(local_zip)
        print(f"using local zip {zip_path}")
    else:
        zip_path = dest_dir.parent / f"{zip_name}.zip"
        url = f"https://datashare.ed.ac.uk/bitstreams/{DATASET_BITSTREAMS[zip_name]}/download"
        print(f"downloading {url} -> {zip_path}")
        _fetch(url, zip_path)

    if not zipfile.is_zipfile(zip_path):
        if not local_zip:
            zip_path.unlink()
        raise RuntimeError(f"{zip_path} is not a valid zip file")
    print(f"extracting {zip_path} -> {dest_dir.parent}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir.parent)
    if not local_zip:
        zip_path.unlink()  # several GB — don't keep the zip alongside the extracted wavs


def ensure_dataset(data_dir: Path, train_zip: str | None = None, test_zip: str | None = None) -> tuple[Path, Path]:
    train_dir = data_dir / "clean_trainset_28spk_wav"
    test_dir = data_dir / "clean_testset_wav"
    try:
        download_and_extract("clean_trainset_28spk_wav", train_dir, train_zip)
        download_and_extract("clean_testset_wav", test_dir, test_zip)
        return train_dir, test_dir
    except Exception as e:
        print(f"dataset download failed ({e!r}); falling back to toy synthetic data.")
        return None, None


def build_channel(kind: str, snr_db: float, rician_k: float) -> ChannelLayer:
    return ChannelLayer(kind, snr_db=snr_db, rician_k=rician_k)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--train-zip", default=None, help="pre-downloaded clean_trainset_28spk_wav.zip, skips the network fetch")
    p.add_argument("--test-zip", default=None, help="pre-downloaded clean_testset_wav.zip, skips the network fetch")
    p.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints"))
    p.add_argument("--subset-size", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--depth", type=int, default=8, help="channel-encoder output depth (compression knob)")
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--channel", choices=["awgn", "rayleigh", "rician"], default="awgn")
    p.add_argument("--rician-k", type=float, default=1.0)
    p.add_argument("--snr-low", type=float, default=0.0)
    p.add_argument("--snr-high", type=float, default=20.0)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--project", default="deepsc-s")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # MPS's autograd breaks on the channel layer's torch.complex ops (see docs);
    # CUDA works fine, CPU is the safe fallback.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    api_key = load_env_key("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    else:
        os.environ.setdefault("WANDB_MODE", "disabled")
        print("no WANDB_API_KEY in .env or environment — W&B logging disabled, continuing without it.")
    run = wandb.init(project=args.project, name=args.run_name, config=vars(args))

    train_dir, test_dir = ensure_dataset(Path(args.data_dir), args.train_zip, args.test_zip)
    if train_dir is not None:
        full_train = SpeechDataset(train_dir, train=True, seed=args.seed)
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(full_train), size=min(args.subset_size, len(full_train)), replace=False)
        train_ds = torch.utils.data.Subset(full_train, idx.tolist())
        test_ds = SpeechDataset(test_dir, train=False, seed=args.seed + 1) if test_dir.exists() else None
    else:
        train_ds = ToyDataset(n=min(args.subset_size, 64), length=W, seed=args.seed)
        test_ds = ToyDataset(n=8, length=W, seed=args.seed + 1)

    print(f"train clips: {len(train_ds)}")
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = DeepSC_S(depth=args.depth, n_blocks=args.n_blocks).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    channel = build_channel(args.channel, args.snr_low, args.rician_k)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in loader:
            batch = batch.to(device).view(-1, 1, F, L)
            channel.snr_db = float(np.random.uniform(args.snr_low, args.snr_high))
            opt.zero_grad()
            x_hat = model(batch, channel=channel)
            loss = loss_fn(x_hat, batch)
            loss.backward()
            opt.step()
            running += loss.item()
        epoch_loss = running / len(loader)
        print(f"epoch {epoch:>4d}  loss={epoch_loss:.5f}")
        wandb.log({"epoch": epoch, "train_loss": epoch_loss})

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            ckpt_path = ckpt_dir / f"deepsc_s_{args.channel}_epoch{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)

    final_path = ckpt_dir / f"deepsc_s_{args.channel}_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"saved final weights to {final_path}")

    if test_ds is not None:
        from deepscs.audio import crop_or_pad, deframe, sdr

        model.eval()
        snr_probe = [0, 10, 20]
        table = wandb.Table(columns=["snr_db", "mean_sdr_db"])
        for snr in snr_probe:
            channel.snr_db = snr
            sdrs = []
            for i in range(min(8, len(test_ds))):
                clip = test_ds[i]
                clip = clip.numpy() if torch.is_tensor(clip) else clip
                xc = torch.from_numpy(crop_or_pad(clip, W)).view(1, 1, F, L).to(device)
                with torch.no_grad():
                    x_hat = model(xc, channel=channel)
                orig = deframe(xc[0, 0].cpu().numpy())
                recon = deframe(x_hat[0, 0].cpu().numpy())
                sdrs.append(sdr(orig, recon))
            mean_sdr = float(np.mean(sdrs))
            print(f"eval SNR={snr:>4} dB -> mean SDR={mean_sdr:.2f} dB")
            table.add_data(snr, mean_sdr)
        wandb.log({"eval_sdr_vs_snr": table})

    run.finish()


if __name__ == "__main__":
    main()
