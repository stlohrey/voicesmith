import subprocess
import os

def run_conda_in_env(command):
    full_command = f"conda run -n aligner {command}"
    subprocess.run(full_command, shell=True, executable="/bin/bash")
    print("been here")



