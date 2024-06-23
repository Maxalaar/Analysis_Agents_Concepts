import subprocess


def remove_conda_environment(env_name):
    try:
        subprocess.run(['conda', 'env', 'remove', '--name', env_name], check=True)
        print(f'The existing environment "{env_name}" has been successfully removed.')
    except subprocess.CalledProcessError as error:
        print(f'An error occurred while removing the environment "{env_name}": {str(error)}.')


def get_environment_conda_name(configuration_path):
    env_name = None
    with open(configuration_path, 'r') as file:
        for line in file:
            if line.startswith('name:'):
                env_name = line.split(':')[1].strip()
                break
    return env_name
