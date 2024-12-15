# -*- coding: utf-8 -*-
# Copyright (C) 2024 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Support functions for a dynamic package version."""


import os
import re
import subprocess
from typing import Optional


def current_git_hash() -> Optional[str]:
    """Get current short git hash.

    Returns:
       Short git hash of current commit, or ``None`` if no git repo found.
    """
    try:
        # See https://stackoverflow.com/questions/14989858
        git_hash: Optional[str] = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.STDOUT,
            )
            .strip()
            .decode()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = None
    if git_hash == "":
        git_hash = None
    return git_hash


def local_version_label(public_version: str) -> str:
    """Get local version label of package version.

    Return an empty string if :code:`public_version` corresponds
    to a release version, otherwise return a local version label
    derived from the current git hash.

    Args:
        public_version: Public version identifier component of
           the full (possibly local) version identifier. (See
           `PEP 440 <https://peps.python.org/pep-0440/>`__.)

    Returns:
        Local version label component of the version identifier.
    """
    # don't extend purely numeric version numbers, possibly ending with rc<n> or post<n>
    if re.match(r"^[0-9\.]+((rc|post)[0-9]+)?$", public_version):
        git_hash = None
    else:
        git_hash = current_git_hash()
    git_hash = "+" + git_hash if git_hash else ""
    return git_hash
