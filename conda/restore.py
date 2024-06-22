import subprocess
import os


def remove_conda_environment(env_name):
    try:
        subprocess.run(['conda', 'env', 'remove', '--name', env_name], check=True)
        print(f'The existing environment "{env_name}" has been successfully removed.')
    except subprocess.CalledProcessError as e:
        print(f'An error occurred while removing the environment "{env_name}": {str(e)}.')


def restore_conda_environment(configuration_path):
    try:
        env_name = None
        with open(configuration_path, 'r') as file:
            for line in file:
                if line.startswith('name:'):
                    env_name = line.split(':')[1].strip()
                    break

        if env_name is not None:
            remove_conda_environment(env_name)

        subprocess.run(['conda', 'env', 'create', '--file', configuration_path], check=True)
        print(f'The environment has been successfully restored from {configuration_path}.')
    except subprocess.CalledProcessError as e:
        print(f'An error occurred while restoring the environment: {str(e)}.')
    except Exception as e:
        print(f'An unexpected error occurred: {str(e)}')


if __name__ == '__main__':
    backup_path = './conda/configuration.yml'
    restore_conda_environment(backup_path)

