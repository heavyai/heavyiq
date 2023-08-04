#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place heavyiq tests --exclude=__init__.py
isort heavyiq tests --profile black
black heavyiq tests --line-length 120
