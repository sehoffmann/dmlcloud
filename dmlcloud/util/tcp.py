import socket


def find_free_port():
    """
    Returns a free port on the local machine.
    """
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def get_local_ips():
    """
    Returns the IP addresses of the local machine.
    """
    hostname = socket.gethostname()
    return socket.gethostbyname_ex(hostname)[2]
