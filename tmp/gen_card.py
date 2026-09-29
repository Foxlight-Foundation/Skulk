"""
Generates inference model cards from Hugging Face metadata, into the current
directory. Skulk ships no cards: curated cards belong in the model registry's
seed, so move a generated card there after review.
Usage:
    uv run tmp/gen_card.py mlx-community/my_cool_model-8bit [repo-id/model-id-2] [...]

Model Cards require cleanup for family & quantization data
"""

import sys

import anyio

from skulk.shared.models.model_cards import ModelCard, ModelId


async def main():
    if len(sys.argv) == 1:
        print(f"USAGE: {sys.argv[0]} repo-id/model-id-1 [repo-id/model-id-2] [...]")
        quit(1)
    print("Remember! Model Cards require cleanup for family & quantization data")
    for arg in sys.argv[1:]:
        mid = ModelId(arg)
        mc = await ModelCard.fetch_from_hf(mid)
        await mc.save((await anyio.Path.cwd()) / (mid.normalize() + ".toml"))


if __name__ == "__main__":
    anyio.run(main)
