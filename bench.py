import os
import pathlib
import subprocess

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tensor_gen import MTX_DIR, VEC_DIR, dense_vector_gen, sparse_matrix_gen

BASE_PATH = pathlib.Path(__file__).resolve().parent

BUILD_DIR = os.path.join(BASE_PATH, "build")
if not os.path.exists(BUILD_DIR):
    os.mkdir(BUILD_DIR)
subprocess.check_output(["cmake", ".."], cwd=BUILD_DIR)
subprocess.check_output(["make"], cwd=BUILD_DIR)

side = [1000, 2000, 5000, 10000, 20000, 30000, 40000]
density = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

BENCHMARK_FREQ = 100

def gen_matrices():
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
                if s == 40000 and d > 80:
                    continue
                output = subprocess.run(["taskset", "-a", "-c", "0", f"./naive_spmv", os.path.join(MTX_DIR, f"{s}x{s}_{d}.mtx"), str(1), str(BENCHMARK_FREQ), os.path.join(VEC_DIR, f"generated_vector_{s}.vector")], cwd=BUILD_DIR, check=True, capture_output=True, text=True)
                exec_time = float(output.stdout.split(" ")[1])
                f.write(f"{s},{d},{exec_time}\n")
                f.flush()

def bench_mkl():
    with open(os.path.join(BASE_PATH, "mkl_spmv.csv"), "w") as f:
        f.write("side,density,time\n")
        for s in side:
            for d in density:
                if s == 40000 and d > 80:
                    continue
                output = subprocess.run(["taskset", "-a", "-c", "0", f"./mkl_spmv", os.path.join(MTX_DIR, f"{s}x{s}_{d}.mtx"), str(1), str(BENCHMARK_FREQ), os.path.join(VEC_DIR, f"generated_vector_{s}.vector")], cwd=BUILD_DIR, check=True, capture_output=True, text=True)
                exec_time = float(output.stdout.split(" ")[1])
                f.write(f"{s},{d},{exec_time}\n")
                f.flush()

def bench_dense():
    with open(os.path.join(BASE_PATH, "dense_spmv.csv"), "w") as f:
        f.write("side,density,time\n")
        for s in side:
            for d in density:
                if s == 40000 and d > 80:
                    continue
                output = subprocess.run(["taskset", "-a", "-c", "0", f"./dense_spmv", os.path.join(MTX_DIR, f"{s}x{s}_{d}.mtx"), str(1), str(BENCHMARK_FREQ), os.path.join(VEC_DIR, f"generated_vector_{s}.vector")], cwd=BUILD_DIR, check=True, capture_output=True, text=True)
                exec_time = float(output.stdout.split(" ")[1])
                f.write(f"{s},{d},{exec_time}\n")
                f.flush()

def draw_heatmap():
    data = pd.read_csv(os.path.join(FILEPATH,"threshold_results.csv"))
    data = data[data['sable_time'] != 0]
    # Calculate the ratio of nonzeros_time to sable_time
    data["time_ratio"] = data[f"CSR_time"] / data["sable_time"]
    # Create a heatmap
    plt.figure(figsize=(8, 6))
    heatmap_data = data.pivot(index="perc_zeros", columns="dim", values="time_ratio")
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={"label": f"Speedup of SABLE dense over SABLE sparse"}
    )
    plt.xlabel("Side of nnz_block")
    plt.ylabel("Percentage of Zeros per nnz_block")
    plt.savefig(os.path.join(FILEPATH,"heatmap.png"))

def draw_heatmap(f_baseline, f_dense):
    baseline_df = pd.read_csv(f_baseline)
    dense_df = pd.read_csv(f_dense)
    speedup = baseline_df['time']/dense_df['time']
    # Plot heatmap of speedup against side and density
    # Ensure necessary columns exist
    if 'side' not in baseline_df or 'density' not in baseline_df:
        raise ValueError("Missing 'side' or 'density' columns in input files")

    # Create a new dataframe with speedup values
    heatmap_data = pd.DataFrame({'side': baseline_df['side'], 
                                 'density': baseline_df['density'], 
                                 'speedup': speedup})

    # Pivot the dataframe for heatmap format
    heatmap_pivot = heatmap_data.pivot(index='side', columns='density', values='speedup')

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_pivot, annot=True, cmap='coolwarm', fmt=".2f")

    # Extract filename from input files
    filename = os.path.basename(f_baseline).split('_')[0]

    plt.title(f"Speedup of dense over {filename}")
    plt.xlabel("Density")
    plt.ylabel("Side Length")
    plt.savefig(os.path.join(BASE_PATH, f"speedup_{filename}.pdf"))

if __name__ == "__main__":
    gen_matrices()
    bench_naive()
    bench_dense()
    bench_mkl()
    naive_path = os.path.join(BASE_PATH, "naive_spmv.csv")
    dense_path = os.path.join(BASE_PATH, "dense_spmv.csv")
    mkl_path = os.path.join(BASE_PATH, "mkl_spmv.csv")
    draw_heatmap(naive_path, dense_path)
    draw_heatmap(mkl_path, dense_path)
