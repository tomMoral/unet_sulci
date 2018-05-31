import os
import numpy as np
import subprocess
import contextlib
import re
import warnings
import pathlib
from functools import wraps


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


@contextlib.contextmanager
def _go_to_repo_directory():
    start_dir = os.path.abspath(os.path.expanduser(os.getcwd()))
    git_repo = str(pathlib.Path(__file__).parents[1])
    os.chdir(git_repo)
    try:
        yield
    finally:
        os.chdir(start_dir)


def _in_repo_directory(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        with _go_to_repo_directory():
            return func(*args, **kwargs)

    return wrapped


@_in_repo_directory
def get_commit_hash():
    resp = None
    # only in python 3.5
    # resp = subprocess.run("git rev-parse HEAD", shell=True,
    #                       stdout=subprocess.PIPE)
    with subprocess.Popen(
            "git rev-list --max-count 1 HEAD", shell=True,
            stdout=subprocess.PIPE) as proc:
        resp = proc.stdout.read()
    commit = re.match(r"^b?([\'\"]?)\b(\w*)\b(.*)\1$", str(resp))
    if commit is not None:
        return commit.group(2)
    warnings.warn('Could not figure out last commit hash')
    return 'UNKNOWN'
