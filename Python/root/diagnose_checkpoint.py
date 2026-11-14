
from pathlib import Path
import gc
import torch

folder = Path("core/ai/dqn/checkpoints")

for path in sorted(folder.glob("normandy_*.pt")):
    print(f"\nChecking {path.name}...", flush=True)

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        for key in (
            "games_completed",
            "total_games",
            "game_count",
        ):
            if key in checkpoint:
                print(f"  {key}: {checkpoint[key]:,}")

        del checkpoint
        gc.collect()

    except Exception as e:
        print(f"  ERROR: {e}")
