from __future__ import annotations

# Receptive field study (theoretical).
#
# We compute:
# - receptive field size r
# - effective stride (jump) j
#
# For each layer with kernel k, stride s:
#     r_new = r + (k - 1) * j
#     j_new = j * s
# Padding does not change receptive field size, it changes alignment.

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class LayerSpec:
    name: str
    k: int
    s: int


def receptive_field(layers: List[LayerSpec]) -> List[Tuple[str, int, int]]:
    r, j = 1, 1  # start at input pixel: RF=1, jump=1
    out = []
    for L in layers:
        r = r + (L.k - 1) * j
        j = j * L.s
        out.append((L.name, r, j))
    return out


def main():
    # matches the default NumPy CNN in train_mnist_numpy.py:
    layers = [
        LayerSpec("conv3x3 s1", 3, 1),
        LayerSpec("pool2x2 s2", 2, 2),
        LayerSpec("conv3x3 s1", 3, 1),
        LayerSpec("pool2x2 s2", 2, 2),
    ]
    table = receptive_field(layers)

    print("Layer\t\tRF size\tJump(stride)")
    for name, r, j in table:
        print(f"{name:12s}\t{r:3d}\t\t{j:3d}")

    last = table[-1]
    print("\nInterpretation:")
    print(f"After the final pooling, each unit 'sees' a {last[1]}x{last[1]} patch of the input image.")


if __name__ == "__main__":
    main()
