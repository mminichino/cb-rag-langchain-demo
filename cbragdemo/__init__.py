import os
from pkg_resources import parse_version

_ROOT = os.path.abspath(os.path.dirname(__file__))
__version__ = "1.0.1"
VERSION = parse_version(__version__)


def get_image_dir():
    return os.path.join(_ROOT, 'images')


def get_image_path(name: str):
    return os.path.join(get_image_dir(), name)
