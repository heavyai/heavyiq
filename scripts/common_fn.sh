# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function process_args() {
  if (( $# == 0 )); then
    return
  fi

  echo "Error. HeavyIQ build artifacts do not package Python dependencies." >&2
  echo "Dependency sources must be configured when requirements.txt is installed." >&2
  echo "Usage: bash scripts/build_prod.sh" >&2
  return 2
}

function prepare_dist() {
  rm -rf ./dist
  mkdir -p ./dist
  cp requirements.txt ./dist/requirements.txt
}

function copy_source_to_dist() {
  if [[ ! -d ./dist ]]; then
    echo "Error. dist directory must exist before copying source" >&2
    return 1
  fi

  cp -r ./heavyiq ./dist/heavyiq
  cp -r ./heavyrag ./dist/heavyrag
  cp gunicorn.conf.py ./dist/gunicorn.conf.py
}

function write_version_to_dist() {
  mkdir -p ./dist/public
  local current_timestamp
  local commit_hash
  current_timestamp=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
  commit_hash=$(git rev-parse --short HEAD)
  echo "${current_timestamp}-${commit_hash}" > ./dist/public/version.txt
}

function verify_no_packaged_dependencies() {
  local dependency_artifact
  dependency_artifact=$(find ./dist \
    \( -type d -name packages -o -type f -name '*.whl' \
       -o -type f -name requirements.packages.txt \) \
    -print -quit)
  if [[ -n $dependency_artifact ]]; then
    echo "Error. Refusing to package dependency artifact: $dependency_artifact" >&2
    return 1
  fi
}

function create_dist_archive() {
  verify_no_packaged_dependencies || return
  (
    cd ./dist
    tar -czf ../dist.tgz .
  )
}
