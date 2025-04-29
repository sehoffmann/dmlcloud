import dmlcloud.core.logging as dml_logging
from dmlcloud.core.callbacks import Callback
from dmlcloud.core.distributed import is_root
from dmlcloud.git import git_diff


class GitDiffCallback(Callback):
    """
    A callback that prints a git diff and if checkpointing is enabled, saves it to the checkpoint directory.
    """

    def pre_run(self, pipe):
        diff = git_diff()
        if diff is None:
            return

        if pipe.run_dir and is_root():
            self._save(pipe.run_dir / 'git_diff.txt', diff)

        msg = '* GIT-DIFF:\n'
        msg += '\n'.join('    ' + line for line in diff.splitlines())
        dml_logging.info(msg)

    def _save(self, path, diff):
        with open(path, 'w') as f:
            f.write(diff)
