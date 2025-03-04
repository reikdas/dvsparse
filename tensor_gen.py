import os
import pathlib
from random import sample

BASE_PATH = pathlib.Path(__file__).resolve().parent

MTX_DIR = os.path.join(BASE_PATH, "Generated_Sparse_Matrices")
VEC_DIR = os.path.join(BASE_PATH, "Generated_dense_tensors")

def sparse_matrix_gen(rows: int, cols: int, density: float, val: float = 1.1) -> None:
    size: int = rows * cols
    nnzs: int = int(size * (density/100))
    indices: list = sample(range(size), nnzs)
    if not os.path.exists(MTX_DIR):
        os.makedirs(MTX_DIR)
    with open(os.path.join(MTX_DIR, f"{rows}x{cols}_{density}.mtx"), "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{rows} {cols} {nnzs}\n")
        for i in indices:
            f.write(f"{(i // cols)+1} {(i % cols)+1} {val}\n")

def dense_vector_gen(size: int, val: float = 1.0):
    filename = f"generated_vector_{size}.vector"
    if not os.path.exists(VEC_DIR):
        os.makedirs(VEC_DIR)
    with open(os.path.join(VEC_DIR, filename), "w") as f:
        x = [val] * size
        f.write(f"{','.join(map(str, x))}\n")
