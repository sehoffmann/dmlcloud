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

    class RedirectedStream:
        def __init__(self, io_redirector, stream_name):
            self.__io_redirector = io_redirector
            self.__stream_name = stream_name
            self.__org_stream = None

        @property
        def __file(self):
            return self.__io_redirector.file

        @property
        def __current_stream(self):
            return getattr(sys, self.__stream_name)

        def install(self):
            self.__org_stream = getattr(sys, self.__stream_name)
            setattr(sys, self.__stream_name, self)

        def uninstall(self):
            setattr(sys, self.__stream_name, self.__org_stream)
            self.__org_stream = None

        def write(self, data):
            if self.__file is not None:
                self.__file.write(data)

            if self.__current_stream is self:  # Avoid infinite recursion
                self.__org_stream.write(data)
            else:
                self.__current_stream.write(data)

        def flush(self):
            if self.__file is not None:
                self.__file.flush()

            if self.__current_stream is self:  # Avoid infinite recursion
                self.__org_stream.flush()
            else:
                self.__current_stream.flush()

        def __getattr__(self, name):
            if self.__current_stream is self:
                return getattr(self.__org_stream, name)
            else:
                raise AttributeError(obj=self, name=name)


    def __init__(self, log_file: Path):
        self.path = log_file
        self.file = None
        self.stdout = None
        self.stderr = None

    def install(self):
        if self.file is not None:
            return

        self.file = self.path.open('a', encoding='utf-8', errors='replace')

        self.stdout = self.RedirectedStream(self, 'stdout')
        self.stdout.install()

        self.stderr = self.RedirectedStream(self, 'stderr')
        self.stderr.install()

    def uninstall(self):
        if self.file is None:
            raise ValueError('IORedirector is not installed')

        self.stdout.uninstall()
        self.stderr.uninstall()

        file = self.file
        self.file = None  # Prevent further writes
        file.close()

        self.stdout = None
        self.stderr = None

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
