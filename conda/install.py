import subprocess


def install_command(command):
    try:
        print('-- --')
        print(f'Running command: {command}')
        subprocess.run(command.split(), check=True)
        print(f'Successfully ran command: {command}')
        print()
    except subprocess.CalledProcessError as error:
        print(f'An error occurred while running command: {command}. Error: {str(error)}')


if __name__ == '__main__':
    # List of commands to run
    commands = [
        "conda install -c conda-forge ray-core --yes",
        "conda install -c conda-forge ray-default --yes",
        "conda install -c conda-forge ray-data --yes",
        "conda install -c conda-forge ray-train --yes",
        "conda install -c conda-forge ray-tune --yes",
        "conda install -c conda-forge ray-rllib --yes",
        "conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia --yes",
        "conda install conda-forge::pytorch-lightning --yes",
        "conda install h5py --yes",
        "conda install -c conda-forge tensorboard --yes",
    ]

    # Execute conda commands
    for command in commands:
        install_command(command)
