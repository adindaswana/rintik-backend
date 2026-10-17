"""Unduh checkpoint + data resmi RINTIK dari repo Hugging Face privat ke RINTIK_ASSET_DIR.

Dipakai saat start di Hugging Face Spaces. Butuh env:
  RINTIK_ASSET_REPO  mis. "nama-akun/rintik-assets" (repo dataset privat)
  HF_TOKEN           token baca (disimpan sebagai Space secret, bukan di kode)
Verifikasi integritas tetap dilakukan oleh inference.py (SHA256 terkunci).
"""
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

FILES = [
    "NSC26-DS011_HBD_File Data Science.pth",
    "train.csv",
    "test.csv",
    "synthetic_physical_humidity_SENSOR_SIMULATION.csv",
    "SOURCE_PROVENANCE.json",
]


def main() -> None:
    repo = os.environ["RINTIK_ASSET_REPO"]
    token = os.environ.get("HF_TOKEN")
    target = Path(os.environ.get("RINTIK_ASSET_DIR", "assets"))
    target.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = target / name
        if dest.is_file():
            continue
        path = hf_hub_download(repo_id=repo, repo_type="dataset", filename=name, token=token)
        shutil.copyfile(path, dest)
        print(f"asset siap: {name} ({dest.stat().st_size} byte)")


if __name__ == "__main__":
    main()
