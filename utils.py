import os
import subprocess

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

BUILD_DIR = os.path.join(BASE_PATH, "build")

def build_project():
    if not os.path.exists(BUILD_DIR):
        os.mkdir(BUILD_DIR)
    subprocess.check_output(["cmake", ".."], cwd=BUILD_DIR)
    subprocess.check_output(["make"], cwd=BUILD_DIR)
