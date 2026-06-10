#!/bin/sh -e
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place heavyiq tests --exclude=__init__.py
isort heavyiq tests --profile black
black heavyiq tests --line-length 120
