"""
This is a small collection of helper functions to debug the installation / imports.

"""

def main():


    import subprocess
    import os
    #Script to check if all requirements are installed
    to_check = {}
    #to_check['java'] = (['Java', '-version'], None) #command, desired output
    to_check['R'] = (['R', '--version'], None)
    #to_check['R path'] = (['which', 'R'], '/usr/local/bin/R')
    to_check['Python'] = (['python', '--version'], None)

    for _ in to_check:
        try:
            out = subprocess.check_output(to_check[_][0], stderr=subprocess.STDOUT).decode("utf-8").rstrip()
            if to_check[_][1] is not None:
                assert to_check[_][1] == out
            print(f"{'='*10} Checking {_} {'='*10} \n{out}")
        except FileNotFoundError as e:
            print(f"Error checking {_}. Error: {e}.")


    # Run all scripts and check if the inputs are correct.

    print('Checking Scripts')

    for path, subdirs, files in os.walk('ckg'):
        for name in files:
            if name.endswith('.py') and not name.startswith('__'):
                path_ = os.path.join(path, name)

                try:
                    out = subprocess.check_output(['python', path_], stderr=subprocess.STDOUT).decode("utf-8").rstrip()
                    print('.')
                except Exception as e:
                    print('\n')
                    print(f"Error: {e}")
