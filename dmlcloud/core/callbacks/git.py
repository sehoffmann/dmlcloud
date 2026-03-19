import dmlcloud.core.logging as dml_logging
from dmlcloud.core.callbacks import Callback
from dmlcloud.core.distributed import is_root
from dmlcloud.git import git_diff


class GitDiffCallback(Callback):
    """
    A callback that prints a git diff and if checkpointing is enabled, saves it to the checkpoint directory.
    """

    def __init__(self, max_chars=2500):
        """
        Args:
            max_chars: The maximum number of characters to log to console. (default: 2500)
        """
        self.max_chars = max_chars

    def _log_diff(self, diff):
        truncate = self.max_chars and len(diff) > self.max_chars

        if truncate:
            msg = '* GIT-DIFF (truncated):\n'
            lines = diff[: self.max_chars].splitlines()
        else:
            msg = '* GIT-DIFF:\n'
            lines = diff.splitlines()

        msg += '\n'.join('    ' + line for line in lines)
        if truncate:
            msg += f'\n    ... (truncated to {self.max_chars} characters, total: {len(diff)})'

        dml_logging.info(msg)

    def pre_run(self, pipe):
        diff = git_diff()
        if diff is None:
            return

        if pipe.run_dir and is_root():
            self._save(pipe.run_dir / 'diagnostics' / 'git_diff.txt', diff)

        self._log_diff(diff)

    def _save(self, path, diff):
        with open(path, 'w') as f:
            f.write(diff)
