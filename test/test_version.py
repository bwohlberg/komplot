# -*- coding: utf-8 -*-
# Copyright (C) 2026 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the komplot package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Tests for module komplot._version."""

import re
import subprocess

import pytest

from komplot import _version


class TestCurrentGitHash:
    """Tests for function current_git_hash."""

    def test_hash_format(self):
        """A hash, if available, should be a short hex string."""
        git_hash = _version.current_git_hash()
        if git_hash is not None:
            assert re.match(r"^[0-9a-f]{7,40}$", git_hash)

    def test_no_git_executable(self, monkeypatch):
        """A missing git executable should be handled gracefully."""

        def raise_not_found(*args, **kwargs):
            raise FileNotFoundError

        monkeypatch.setattr(subprocess, "check_output", raise_not_found)
        assert _version.current_git_hash() is None

    def test_not_a_repository(self, monkeypatch):
        """A failed git call should be handled gracefully."""

        def raise_called_process_error(*args, **kwargs):
            raise subprocess.CalledProcessError(128, "git")

        monkeypatch.setattr(subprocess, "check_output", raise_called_process_error)
        assert _version.current_git_hash() is None

    def test_empty_hash(self, monkeypatch):
        """Empty git output should be mapped to ``None``."""
        monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: b"  \n")
        assert _version.current_git_hash() is None


class TestLocalVersionLabel:
    """Tests for function local_version_label."""

    @pytest.mark.parametrize(
        "public_version", ["0", "1.2", "0.0.5", "0.0.5rc1", "0.0.5post2"]
    )
    def test_release_version(self, public_version, monkeypatch):
        """Release versions should have an empty local version label."""
        monkeypatch.setattr(_version, "current_git_hash", lambda: "abcdef1")
        assert _version.local_version_label(public_version) == ""

    @pytest.mark.parametrize(
        "public_version", ["0.0.5.dev0", "0.0.5dev", "0.0.5+local"]
    )
    def test_development_version(self, public_version, monkeypatch):
        """Development versions should be labelled with the git hash."""
        monkeypatch.setattr(_version, "current_git_hash", lambda: "abcdef1")
        assert _version.local_version_label(public_version) == "+abcdef1"

    def test_development_version_without_hash(self, monkeypatch):
        """An unavailable git hash should give an empty local version label."""
        monkeypatch.setattr(_version, "current_git_hash", lambda: None)
        assert _version.local_version_label("0.0.5.dev0") == ""
