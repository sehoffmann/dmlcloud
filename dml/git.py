import subprocess
from pathlib import Path


def run_in_project(cmd):
    project_dir = Path(__file__).absolute().parent
    return subprocess.run(cmd, cwd=project_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def git_hash(short=False):
    if short:
        process = run_in_project(['git', 'rev-parse', '--short', 'HEAD'])
    else:
        process = run_in_project(['git', 'rev-parse', 'HEAD'])
    return process.stdout.decode('utf-8').strip()


def git_diff():
    process = run_in_project(['git', 'diff', '-U0', '--no-color', 'HEAD'])
    return process.stdout.decode('utf-8').strip()
