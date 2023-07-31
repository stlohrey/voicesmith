import subprocess
import os
def run_conda_in_env(command):
    # Get the path to the Conda executable
    conda_path = subprocess.check_output(["which", "conda"], text=True).strip()
    environment_name=os.environ['CONDA_DEFAULT_ENV']
    #print(environment_name)
    # Run the desired command within the specified Conda environment
    subprocess.run([conda_path, "run", "-n", environment_name, "bash", "-c", command], check=True)

