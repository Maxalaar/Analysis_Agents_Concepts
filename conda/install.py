import subprocess


def install_conda_package(command):
    try:
        print(f'Running command: {command}')
        subprocess.run(command.split(), check=True)
        print(f'Successfully ran command: {command}')
    except subprocess.CalledProcessError as error:
        print(f'An error occurred while running command: {command}. Error: {str(error)}')


def install_pip_package(package_name):
    try:
        print(f'Installing or updating pip package: {package_name}')
        subprocess.run(['pip', 'install', package_name], check=True)
        print(f'Successfully installed or updated pip package: {package_name}')
    except subprocess.CalledProcessError as error:
        print(f'An error occurred while installing or updating pip package: {package_name}. Error: {str(error)}')


if __name__ == '__main__':
    # List of conda commands to run
    conda_commands = [
        "conda install -c conda-forge ray-default",
        "conda install -c conda-forge ray-data",
        "conda install -c conda-forge ray-train",
        "conda install -c conda-forge ray-tune",
        "conda install -c conda-forge ray-serve",
        "conda install -c conda-forge ray-rllib",
        "conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia",
        "conda install conda-forge::pytorch-lightning ",
    ]

    # List of pip packages to install
    pip_packages = [
    ]

    # Execute conda commands
    for conda_command in conda_commands:
        install_conda_package(conda_command)

    # Install pip packages
    for package in pip_packages:
        install_pip_package(package)
