import io
import sys
from datetime import timedelta
from pathlib import Path


class TimedeltaFormatter:
    """
    A formatter that converts a number of seconds to a human-readable string.
    """

    def __init__(self, microseconds=False):
        self.microseconds = microseconds

    def __call__(self, seconds: float) -> str:
        delta = timedelta(seconds=seconds)
        if not self.microseconds:
            delta -= timedelta(microseconds=delta.microseconds)
        return str(delta)


class IORedirector:
    """
    Context manager to redirect stdout and stderr to a file.
    Data is written to the file and the original streams.
    """

    # Caveats:
    #  * We always need to forward the current stdout/stderr. People can change them.
    #  * Even after uninstall, people can still hold reference to the redirected streams.
    #    Hence, we must be fault tolorant and not crash if the file is closed or the streams are changed.

    class Stdout:
        def __init__(self, parent):
            self.parent = parent

        def write(self, data):
            if self.parent.file is not None:
                self.parent.file.write(data)

            if sys.stdout is self:  # Avoid infinite recursion
                self.parent._org_stdout.write(data)
            else:
                sys.stdout.write(data)

        def flush(self):
            if self.parent.file is not None:
                self.parent.file.flush()

            if sys.stdout is self:  # Avoid infinite recursion
                self.parent._org_stdout.flush()
            else:
                sys.stdout.flush()

    class Stderr:
        def __init__(self, parent):
            self.parent = parent

        def write(self, data):
            if self.parent.file is not None:
                self.parent.file.write(data)

            if sys.stderr is self:  # Avoid infinite recursion
                self.parent._org_stderr.write(data)
            else:
                sys.stderr.write(data)

        def flush(self):
            if self.parent.file is not None:
                self.parent.file.flush()

            if sys.stderr is self:  # Avoid infinite recursion
                self.parent._org_stderr.flush()
            else:
                sys.stderr.flush()

    def __init__(self, log_file: Path):
        self.path = log_file
        self.file = None
        self._org_stdout = None
        self._org_stderr = None

    def install(self):
        if self.file is not None:
            return

        self.file = self.path.open('a', encoding='utf-8', errors='replace')
        self._org_stdout = sys.stdout
        self._org_stderr = sys.stderr
        self._org_stdout.flush()
        self._org_stderr.flush()

        sys.stdout = self.Stdout(self)
        sys.stderr = self.Stderr(self)

    def uninstall(self):
        if self.file is None:
            raise ValueError('IORedirector is not installed')

        sys.stdout = self._org_stdout
        sys.stderr = self._org_stderr

        file = self.file
        self.file = None  # Prevent further writes
        file.close()

        self._org_stdout = None
        self._org_stderr = None

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.uninstall()


class DevNullIO(io.TextIOBase):
    """
    Dummy TextIOBase that will simply ignore anything written to it similar to /dev/null
    """

    def write(self, msg):
        pass
