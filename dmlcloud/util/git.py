from .project import run_in_project, script_path


def git_hash(short=False):
    if script_path() is None:
        return None
    if short:
        process = run_in_project(['git', 'rev-parse', '--short', 'HEAD'])
    else:
        process = run_in_project(['git', 'rev-parse', 'HEAD'])
    return process.stdout.decode('utf-8').strip()


def git_diff():
    if script_path() is None:
        return None
    process = run_in_project(['git', 'diff', '-U0', '--no-color', 'HEAD'])
    return process.stdout.decode('utf-8').strip()
