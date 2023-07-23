#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place heavynl tests --exclude=__init__.py
isort heavynl tests --profile black
black heavynl tests --line-length 120
