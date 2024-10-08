#!/usr/bin/env python3

from pathlib import Path
from setuptools import setup


if __name__ == '__main__':
    setup(name='custom_env',
          use_scm_version=dict(
              root='..',
              relative_to=__file__,
              version_scheme='no-guess-dev'
          ),
          scripts=[],
          version='1.1'
          )
