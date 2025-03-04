import os
import pathlib
import subprocess

from tensor_gen import MTX_DIR, VEC_DIR, dense_vector_gen, sparse_matrix_gen

BASE_PATH = pathlib.Path(__file__).resolve().parent

BUILD_DIR = os.path.join(BASE_PATH, "build")
if not os.path.exists(BUILD_DIR):
    os.mkdir(BUILD_DIR)
subprocess.check_output(["cmake", ".."], cwd=BUILD_DIR)
subprocess.check_output(["make"], cwd=BUILD_DIR)

def gen_matrices():
    side = [1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000]
    density = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    MAT_VAL = 1.1
    VEC_VAL = 1.5
    for s in side:
        dense_vector_gen(s, VEC_VAL)
        for d in density:
            sparse_matrix_gen(s, s, d, MAT_VAL)

def bench_naive():
    with open(os.path.join(BASE_PATH, "naive_spmv.csv"), "w") as f:
        f.write("side,density,time\n")
    for s in side:
        for d in density:
            output = subprocess.run(["taskset", "-a", "-c", "0", f"./naive_spmv", os.path.join(MTX_DIR, f"{s}x{s}_{d}.mtx"), str(1), str(1), os.path.join(VEC_DIR, f"generated_vector_{s}.vector")], cwd=BUILD_DIR, check=True, capture_output=True, text=True)
            exec_time = float(output.stdout.split(" ")[1])
            f.write(f"{s},{d},{exec_time}\n")
            f.flush()

def bench_mkl():
    with open(os.path.join(BASE_PATH, "mkl_spmv.csv"), "w") as f:
        f.write("side,density,time\n")
    for s in side:
        for d in density:
            output = subprocess.run(["taskset", "-a", "-c", "0", f"./mkl_spmv", os.path.join(MTX_DIR, f"{s}x{s}_{d}.mtx"), str(1), str(1), os.path.join(VEC_DIR, f"generated_vector_{s}.vector")], cwd=BUILD_DIR, check=True, capture_output=True, text=True)
            exec_time = float(output.stdout.split(" ")[1])
            f.write(f"{s},{d},{exec_time}\n")
            f.flush()

def bench_dense():
    with open(os.path.join(BASE_PATH, "dense_spmv.csv"), "w") as f:
        f.write("side,density,time\n")
    for s in side:
        for d in density:
            output = subprocess.run(["taskset", "-a", "-c", "0", f"./dense_spmv", os.path.join(MTX_DIR, f"{s}x{s}_{d}.mtx"), str(1), str(1), os.path.join(VEC_DIR, f"generated_vector_{s}.vector")], cwd=BUILD_DIR, check=True, capture_output=True, text=True)
            exec_time = float(output.stdout.split(" ")[1])
            f.write(f"{s},{d},{exec_time}\n")
            f.flush()

if __name__ == "__main__":
    gen_matrices()
    bench_naive()
    bench_mkl()
    bench_dense()
