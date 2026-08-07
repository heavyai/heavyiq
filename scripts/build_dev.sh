# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ENV Requirements
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source  $SCRIPT_DIR/common_fn.sh
process_args "$@" || exit $?

# Copy to dist
rm -rf dist packages
mkdir -p dist
cp -r heavyiq dist/heavyiq
cp requirements.txt ./dist/requirements.txt

# If requested, store only the selected pyheavydb wheel under dist/packages.
test_for_internally_release_pyheavydb
if [[ -d packages ]]; then
  mv packages dist/.
fi

# Create version.txt
mkdir -p dist/public
current_timestamp=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
commit_hash=$(git rev-parse --short HEAD)
echo "${current_timestamp}-${commit_hash}" > dist/public/version.txt

# Compress Build Dir
tar -czf dist.tgz dist
