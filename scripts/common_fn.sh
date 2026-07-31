# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ENV Requirements
## python3.10 (Installed)

export INTERNALLY_RELEASED_PYHEAVYDB=false
export INCLUDE_ALL_DEPS=false
export HTTP_PYTHON_DEPS="https://dependencies.heavy.ai/python-deps"
export NFS_PATH="/theHoard/export/home/www/dependencies.mapd.com/python-deps"
export PKG_PATH=""
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
      --include_all_deps)
        INCLUDE_ALL_DEPS=true
        ;;
      *)
        break
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

function test_for_include_all_deps() {
  if [[ $INCLUDE_ALL_DEPS == "false" ]]; then
    return
  fi
  local requirements_file=$1
  if [[ -z $requirements_file ]]; then
    echo "Error. requirements_file required as a parameter"
    return
  fi
  local requirements_file_rhel=$2

  ## Assume this happens after the dist dir has been
  ## made and the requirements.txt files has been copied
  if [[ ! -d ./dist ]] ; then
    echo "WARNING. Either the dist dir has not been created or" \
      "the script is being run from the wrong place"
    return
  fi

  ## If a local copy of pyheavydb hasn't already been added then
  ## the ./packages dir will not have been created. Hence mkdir -p
  mkdir -p ./packages

  ## Download from requirements.txt only
  pip download -r $requirements_file --dest ./packages/

  ## For the "rhel" case, fetch the RHEL-specific artifacts straight from
  ## PyPI / Test-PyPI (no vendored binaries or private mirror needed).
  if [[ $requirements_file_rhel == "rhel" ]]; then
    # chromadb ships an abi3 manylinux x86_64 wheel on Test-PyPI. Force it so the
    # correct wheel is used regardless of the build host's OS/arch.
    rm -f ./packages/chromadb-*.whl
    pip download chromadb==1.5.10.dev197 \
      --only-binary=:all: --platform manylinux2014_x86_64 \
      --python-version 39 --abi abi3 --implementation cp \
      --no-deps --dest ./packages \
      --extra-index-url https://test.pypi.org/simple/

    # pysqlite3-binary is not a declared dependency; it provides a modern SQLite
    # that chromadb needs on RHEL 8. Fetch the target's cp311 manylinux wheel.
    pip download pysqlite3-binary==0.5.3 \
      --only-binary=:all: --platform manylinux2014_x86_64 \
      --python-version 311 --abi cp311 --implementation cp \
      --no-deps --dest ./packages

    # PyPI publishes only an sdist for PyPika 0.48.9; build the pure-Python wheel.
    # (pip download -r may have pulled a newer pypika transitively; drop it first.)
    rm -f ./packages/[Pp]y[Pp]ika-*.whl ./packages/[Pp]y[Pp]ika-*.tar.gz
    pip wheel PyPika==0.48.9 --no-deps -w ./packages
  fi

  ## Some dependencies are published only as source distributions (e.g.
  ## pyheavydb ships an sdist). Build them into wheels now, while the network is
  ## available, so the offline deploy install never has to compile or fetch
  ## build backends (which fails under `pip install --no-index`).
  shopt -s nullglob
  for sdist in ./packages/*.tar.gz ./packages/*.zip; do
    pip wheel "$sdist" --no-deps -w ./packages && rm -f "$sdist"
  done
  shopt -u nullglob

  ## Some wheels (e.g. pybase64) stack multiple platform tags, producing a
  ## filename longer than tar's 100-char ustar name limit. Such names rely on
  ## GNU/PAX long-name extensions and can be dropped or mangled when the build
  ## tarball is extracted at deploy time, causing "No such file" install errors.
  ## For any over-long name, collapse the stacked platform tags down to the
  ## first one; every tag in the set is satisfied by our glibc deploy target, so
  ## keeping one is safe. Renaming only changes the advertised filename tags
  ## (pip reads tags from the filename), it does not rebuild the wheel.
  python3 - <<'PY'
import pathlib
import re

pkg = pathlib.Path("./packages")
for whl in pkg.glob("*.whl"):
    name = whl.name
    if len(name) <= 99:
        continue
    new = re.sub(r"(\.[a-z0-9_]+)+\.whl$", ".whl", name)
    target = pkg / new
    if new == name:
        continue
    if target.exists():
        print(f"skipped; target exists: {target}")
        continue
    whl.rename(target)
    print(f"{name} -> {new}")
PY

  ## generate a new requirements file than will use the
  ## archives in the packages directory
  find ./packages -type f > ./dist/requirements.packages.txt
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

# Overrides apply only to generated staging manifests. requirements.txt in the
# project remains the authoritative dependency declaration.
function update_pyheavydb_reference() {
  if [[ $INTERNALLY_RELEASED_PYHEAVYDB == "false" ]]; then
    return
  fi
  local requirements_file=$1
  if [[ -z $requirements_file ]]; then
    echo "Error. requirements_file required as a parameter"
    return
  fi
  if [[ -z $PKG_PATH ]]; then
    echo "Error. Updating [${requirements_file}] without a package definition"
    return
  fi

  # Replace pyheavydb only in the generated build manifest.
  sed -i "/pyheavydb/d" "${requirements_file}"
  sed -i "1 i ${PKG_PATH}" "${requirements_file}"
}

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
  PKG_PATH="./packages/${PYHEAVYDB_ARCHIVE}"
  update_pyheavydb_reference ./dist/requirements.txt
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
