# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ENV Requirements
## python3.10 (Installed)

export INTERNALLY_RELEASED_PYHEAVYDB=false
export HTTP_PYTHON_DEPS="https://dependencies.heavy.ai/python-deps"
export NFS_PATH="/theHoard/export/home/www/dependencies.mapd.com/python-deps"
export PYHEAVYDB_ARCHIVE=""
export PYHEAVYDB_WHEEL=""

function process_args(){
  while (( $# )); do
    case "$1" in
      --internally-built-pyheavydb)
        INTERNALLY_RELEASED_PYHEAVYDB=true
        ;;
      --pyheavydb-wheel=*)
        PYHEAVYDB_WHEEL="${1#*=}"
        INTERNALLY_RELEASED_PYHEAVYDB=true
        ;;
      *)
        echo "Error. Unsupported build option: $1" >&2
        echo "Usage: bash scripts/build_prod.sh [--pyheavydb-wheel=/path/to/pyheavydb.whl]" >&2
        return 2
        ;;
    esac
    shift
  done
}

function get_pyheavydb_for_local_install() {
  wget --continue ${HTTP_PYTHON_DEPS}/pyheavydb.whl.version
  PYHEAVYDB_ARCHIVE=$(head -n 1 pyheavydb.whl.version)
  wget --continue ${HTTP_PYTHON_DEPS}/${PYHEAVYDB_ARCHIVE}
}

function get_pyheavydb_for_NFS_install() {
  pwd
  cp ${NFS_PATH}/pyheavydb.whl.version .
  PYHEAVYDB_ARCHIVE=$(head -n 1 pyheavydb.whl.version)
  cp ${NFS_PATH}/${PYHEAVYDB_ARCHIVE} .
}


function copy_source_to_dist() {
  if [[ ! -d ./dist ]] ; then
    echo "Error. dist directory must exist before copying source"
    return 1
  fi

  rm -rf ./dist/heavyiq ./dist/heavyrag
  cp -r ./heavyiq ./dist/heavyiq
  cp -r ./heavyrag ./dist/heavyrag
  cp gunicorn.conf.py ./dist/gunicorn.conf.py
}

# Package pyheavydb without modifying the project's authoritative requirements.

function test_for_internally_release_pyheavydb() {
  if [[ $INTERNALLY_RELEASED_PYHEAVYDB == "false" ]];then
    return
  fi

  ## Assume this happens after the dist dir has been
  ## made and the requirements.txt files has been copied
  if [[ ! -d ./dist ]] ; then
    echo "WARNING. Either the dist dir has not been created or" \
      "the script is being run from the wrong place"
    return
  fi
  mkdir -p ./packages
  if [[ -n $PYHEAVYDB_WHEEL ]]; then
    if [[ ! -f $PYHEAVYDB_WHEEL ]]; then
      echo "Error. pyheavydb wheel not found: ${PYHEAVYDB_WHEEL}" >&2
      return 1
    fi
    PYHEAVYDB_ARCHIVE=$(basename "$PYHEAVYDB_WHEEL")
    cp "$PYHEAVYDB_WHEEL" "./packages/$PYHEAVYDB_ARCHIVE"
  else
    # Legacy compatibility path for --internally-built-pyheavydb.
    get_pyheavydb_for_NFS_install
    mv "$PYHEAVYDB_ARCHIVE" ./packages
  fi
}

function test_and_install_local_pyheavydb() {
  # get_pyheavydb_for_local_install sets PYHEAVYDB_ARCHIVE
  ##get_pyheavydb_for_local_install
  get_pyheavydb_for_NFS_install
  if [[ $INTERNALLY_RELEASED_PYHEAVYDB == "false" ]];then
    return
  fi
  pip install ${PYHEAVYDB_ARCHIVE}
}
