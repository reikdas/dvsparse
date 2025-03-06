import pathlib
import os
import scipy
import subprocess

import joblib
import tqdm

from utils import build_project, BUILD_DIR
from tensor_gen import VEC_DIR, dense_vector_gen

BASE_PATH = pathlib.Path(__file__).resolve().parent

BENCHMARK_FREQ = 100

def check_file_matches_parent_dir(filepath):
    """
    Check if a file's name (without suffix) matches its parent directory name.
    
    Args:
        filepath (str): Full path to the file
        
    Returns:
        bool: True if file name (without suffix) matches parent directory name
        
    Example:
        >>> path = '/local/scratch/a/das160/SABLE/Suitesparse/GD96_a/GD96_a.mtx'
        >>> check_file_matches_parent_dir(path)
        True
    """
    # Get the file name without extension
    file_name = os.path.splitext(os.path.basename(filepath))[0]
    
    # Get the parent directory name
    parent_dir = os.path.basename(os.path.dirname(filepath))
    
    return file_name == parent_dir

if __name__ == "__main__":
    build_project()
    model = joblib.load(os.path.join(BASE_PATH, "models", "density_threshold_spmv.pkl"))
    mtx_dir = pathlib.Path("/local/scratch/a/Suitesparse")
    with open(os.path.join(BASE_PATH, "results", "suitesparse_output.csv"), "w") as f:
        f.write("Name,Size,Nnz,Density,Speedup_Naive,Speedup_MKL\n")
        for file_path in tqdm.tqdm(mtx_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix == ".mtx" and check_file_matches_parent_dir(file_path):
                fname = file_path.stem
                A = scipy.io.mmread(file_path)
                size = A.shape[0] * A.shape[1]
                density = (A.nnz/size) * 100
                if model.predict([[size, density]])[0] == 1:
                    dense_vector_gen(A.shape[1], val=1.5)
                    output = subprocess.run(["taskset", "-a", "-c", "0", f"./naive_spmv", file_path, str(1), str(BENCHMARK_FREQ), os.path.join(VEC_DIR, f"generated_vector_{A.shape[1]}.vector")], cwd=BUILD_DIR, check=True, capture_output=True, text=True)
                    naive_exec_time = float(output.stdout.split(" ")[1])
                    output = subprocess.run(["taskset", "-a", "-c", "0", f"./dense_spmv", file_path, str(1), str(BENCHMARK_FREQ), os.path.join(VEC_DIR, f"generated_vector_{A.shape[1]}.vector")], cwd=BUILD_DIR, check=True, capture_output=True, text=True)
                    dense_exec_time = float(output.stdout.split(" ")[1])
                    output = subprocess.run(["taskset", "-a", "-c", "0", f"./mkl_spmv", file_path, str(1), str(BENCHMARK_FREQ), os.path.join(VEC_DIR, f"generated_vector_{A.shape[1]}.vector")], cwd=BUILD_DIR, check=True, capture_output=True, text=True)
                    mkl_exec_time = float(output.stdout.split(" ")[1])
                    speedup_naive = naive_exec_time / dense_exec_time
                    speedup_mkl = mkl_exec_time / dense_exec_time
                    f.write(f"{fname},{size},{A.nnz},{density},{speedup_naive},{speedup_mkl}\n")
