import os
import pathlib
import subprocess

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tensor_gen import MTX_DIR, VEC_DIR, dense_vector_gen, sparse_matrix_gen
from utils import build_project, BUILD_DIR

from src.codegen_many_dense import codegen

BASE_PATH = pathlib.Path(__file__).resolve().parent

plt.rcParams['figure.dpi'] = 300

def draw_heatmap():
    df = pd.read_csv(os.path.join(BASE_PATH, "results", "regalloc.csv"))

    # Convert time to ms
    df['Time'] = df['Time']/1_000_000

    # Create a new dataframe with speedup values
    baseline_times = df[df['NumCalls'] == 1].set_index('Rows')['Time']

    # Compute speedup
    df['Speedup'] = df.apply(lambda row: baseline_times.get(row['Rows'], float('nan')) / row['Time'], axis=1)

    # Create a new dataframe with speedup values
    heatmap_data = pd.DataFrame({'Rows': df['Rows'], 
                                 'NumCalls': df['NumCalls'], 
                                 'Speedup': df['Speedup']})

    # Pivot the dataframe for heatmap format
    heatmap_pivot = heatmap_data.pivot(index='Rows', columns='NumCalls', values='Speedup')

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_pivot, annot=True, cmap='coolwarm', fmt=".2f")

    plt.title("Speedup Heatmap")
    plt.xlabel("NumCalls")
    plt.ylabel("Rows")
    plt.savefig(os.path.join(BASE_PATH, "results", "speedup_many.pdf"))

def bench():
    MAT_VAL = 1.1
    VEC_VAL = 1.5
    cols = 51200
    num_calls_list = [1,2,4,8,5,10,20,40,25,50,100,200]
    rows_list = [256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560]
    BENCHMARK_FREQ = 100
    dense_vector_gen(cols, VEC_VAL)
    with open(os.path.join("results", "regalloc.csv"), "w") as f:
        f.write("Rows,NumCalls,Time\n")
        for rows in rows_list:
            sparse_matrix_gen(rows, cols, 100, MAT_VAL)
            for num_calls in num_calls_list:
                codegen(num_calls, os.path.join(MTX_DIR, f"{rows}x{cols}_100.mtx"))
                build_project()
                output = subprocess.run(["taskset", "-a", "-c", "0", f"./many_dense", os.path.join(MTX_DIR, f"{rows}x{cols}_100.mtx"), str(1), str(BENCHMARK_FREQ), os.path.join(VEC_DIR, f"generated_vector_{cols}.vector")], cwd=BUILD_DIR, check=True, capture_output=True, text=True)
                many_exec_time = float(output.stdout.split(" ")[1])
                f.write(f"{rows},{num_calls},{many_exec_time}\n")
                f.flush()
            os.remove(os.path.join(MTX_DIR, f"{rows}x{cols}_100.mtx"))


if __name__ == "__main__":
    bench()
    draw_heatmap()
