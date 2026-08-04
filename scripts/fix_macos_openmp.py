"""Make LightGBM importable on macOS without Homebrew (idempotent).

python.org / pip LightGBM wheels link against `@rpath/libomp.dylib`. When
Homebrew's libomp is absent, `import lightgbm` fails; and if a *separate copy*
of libomp is loaded alongside PyTorch's, the two OpenMP runtimes can crash at
train time. Both are fixed by symlinking LightGBM's expected libomp to the one
PyTorch already ships (so a single OpenMP image is loaded).

    python scripts/fix_macos_openmp.py

No-op on non-macOS or when LightGBM already imports.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _bundled_libomp() -> Path | None:
    import sysconfig
    site = Path(sysconfig.get_paths()["purelib"])
    for cand in (site / "torch" / "lib" / "libomp.dylib",
                 site / "sklearn" / ".dylibs" / "libomp.dylib"):
        if cand.exists():
            return cand
    return None


def main() -> int:
    if sys.platform != "darwin":
        print("Not macOS; nothing to do.")
        return 0
    try:
        import lightgbm  # noqa: F401
        # Force a train to surface a latent OpenMP clash.
        import numpy as np
        import lightgbm as lgb
        d = lgb.Dataset(np.random.rand(50, 3), label=(np.random.rand(50) > .5).astype(int),
                        group=[25, 25])
        lgb.train({"objective": "lambdarank", "verbose": -1}, d, num_boost_round=2)
        print("LightGBM already works.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"LightGBM not usable yet ({type(exc).__name__}); attempting fix...")

    src = _bundled_libomp()
    if src is None:
        print("No bundled libomp found (install torch or scikit-learn, "
              "or `brew install libomp`).")
        return 1

    import lightgbm
    dest_dir = Path(lightgbm.__file__).parent / "lib"
    dest = dest_dir / "libomp.dylib"
    dest.unlink(missing_ok=True)
    os.symlink(src, dest)
    print(f"Linked {dest} -> {src}")
    print("Re-run this script to verify.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
