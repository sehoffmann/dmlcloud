import socket
import subprocess

def find_free_port():
    """
    Returns a free port on the local machine.
    """
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def get_local_ips(use_hostname=True):
    """
    Returns the IP addresses of the local machine.
    """
    if use_hostname:
        proc = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
        if proc.returncode == 0:
            return proc.stdout.strip().split(' ')
        else:
            err = proc.stderr.strip()
            raise RuntimeError(err)
    else:
        hostname = socket.gethostname()
        return socket.gethostbyname_ex(hostname)[2]
