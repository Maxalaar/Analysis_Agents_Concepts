import subprocess


def save_conda_environment(path):
    try:
        subprocess.run(['conda', 'env', 'export', '--no-builds', '--file', path], check=True)
        print('The environment has been successfully saved to ' + path + '.')

        # Remove the prefix line from the exported file
        with open(path, 'r') as file:
            lines = file.readlines()

        with open(path, 'w') as file:
            for line in lines:
                if not line.strip().startswith('prefix:'):
                    file.write(line)

        print('The prefix line has been successfully removed from ' + path + '.')
    except subprocess.CalledProcessError as e:
        print(f'An error occurred while saving the environment: {e}')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')


if __name__ == '__main__':
    backup_path = './conda/configuration.yml'
    save_conda_environment(backup_path)
