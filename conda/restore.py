import subprocess

from utilities import get_environment_conda_name, remove_conda_environment


def restore_conda_environment(configuration_path):
    try:
        environment_name = get_environment_conda_name
        remove_conda_environment(environment_name)

        subprocess.run(['conda', 'env', 'create', '--file', configuration_path], check=True)
        print(f'The environment has been successfully restored from {configuration_path}.')
    except subprocess.CalledProcessError as error:
        print(f'An error occurred while restoring the environment: {str(error)}.')
    except Exception as error:
        print(f'An unexpected error occurred: {str(error)}')


if __name__ == '__main__':
    backup_path = './conda/configuration.yml'
    restore_conda_environment(backup_path)

