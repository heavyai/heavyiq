# ENV Requirements
## python3.10 (Installed)

export INTERNALLY_RELEASED_PYHEAVYDB=false
export INCLUDE_ALL_DEPS=false
export HTTP_PYTHON_DEPS="https://dependencies.mapd.com/python-deps"
export PKG_PATH=""
export PYHEAVYDB_ARCHIVE=""

function process_args(){
  while (( $# )); do
    case "$1" in
      --internally-built-pyheavydb)
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

  ## For the "rhel" case, copy RHEL-specific files directly
  if [[ $requirements_file_rhel == "rhel" ]]; then
    # Replace the downloaded chromadb-0.5.3-py3-none-any.whl with the one in scripts/assets
    cp scripts/assets/chromadb-0.5.3-py3-none-any.whl ./packages/
    cp scripts/assets/pysqlite3_binary-0.5.3-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl ./packages/

    # Replace PyPika with a binary version found in scripts/assets
    rm ./packages/PyPika-0.48.9.tar.gz
    cp scripts/assets/PyPika-0.48.9-py2.py3-none-any.whl ./packages/
  fi

  ## generate a new requirements file than will use the
  ## archives in the packages directory
  find ./packages -type f > ./dist/requirements.packages.txt
}

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

  # 1. If any then remove any existing refence to pyheavydb
  # 2. Add the relative file reference at the top of the file.
  sed -i "/pyheavydb/d" ${requirements_file}
  sed -i "1 i ${PKG_PATH}" ${requirements_file}
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
  # get_pyheavydb_for_local_install sets PYHEAVYDB_ARCHIVE
  get_pyheavydb_for_local_install
  mkdir -p ./packages
  mv ${PYHEAVYDB_ARCHIVE} ./packages
  PKG_PATH=./packages/${PYHEAVYDB_ARCHIVE}
  update_pyheavydb_reference ./dist/requirements.txt
}

function test_and_install_local_pyheavydb() {
  # get_pyheavydb_for_local_install sets PYHEAVYDB_ARCHIVE
  get_pyheavydb_for_local_install
  if [[ $INTERNALLY_RELEASED_PYHEAVYDB == "false" ]];then
    return
  fi
  pip install ${PYHEAVYDB_ARCHIVE}
}
