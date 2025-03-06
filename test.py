import os
import pathlib
import subprocess

import scipy

from tensor_gen import MTX_DIR, VEC_DIR, dense_vector_gen, sparse_matrix_gen
from utils import build_project, BUILD_DIR

BASE_PATH = pathlib.Path(__file__).resolve().parent

def cmp_file(file1, file2):
    with open(file1, "r") as f1, open(file2, "r") as f2:
        for line1, line2 in zip(f1, f2):
            line1 = line1.strip()
            line2 = line2.strip()
            # Attempt to compare as floats if the lines contain numeric data
            try:
                # This will succeed if both lines are numeric
                if float(line1) != float(line2):
                    return False
            except ValueError:
                # If they aren't numeric, compare them as strings
                if line1 != line2:
                    return False
    return True

build_project()

VEC_VAL = 1.5

def setup_file():
    sparse_matrix_gen(rows=3, cols=3, density=90, val=1.1)
    dense_vector_gen(3, val=VEC_VAL)

def run_test(baseline: str):
    setup_file()
    baseline_output = subprocess.run([f"./{baseline}", os.path.join(MTX_DIR, "3x3_90.mtx"), str(1), str(1), os.path.join(VEC_DIR, "generated_vector_3.vector")], capture_output=True, cwd=BUILD_DIR)
    output = baseline_output.stdout.decode("utf-8").split("\n")[1:]
    with open(os.path.join("tests", "output.txt"), "w") as f:
        f.write("\n".join(output))
    mtx = scipy.io.mmread(os.path.join(BASE_PATH, "Generated_Sparse_Matrices", "3x3_90.mtx"))
    vec = [VEC_VAL] * 3
    result = [round(x, 2) for x in mtx @ vec]
    with open(os.path.join("tests", "expected_output.txt"), "w") as f:
        f.write("\n".join(map(str, result)))
    assert cmp_file(os.path.join("tests", "output.txt"), os.path.join("tests", "expected_output.txt"))

def test_naive():
    run_test("naive_spmv")

def test_mkl():
    run_test("mkl_spmv")

def test_dense():
    run_test("dense_spmv")
