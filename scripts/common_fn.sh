# ENV Requirements
## python3.10 (Installed)

export INTERNALLY_RELEASED_PYHEAVYDB=false
export HTTP_PYTHON_DEPS="https://dependencies.mapd.com/python-deps"
export PKG_PATH=""
export PYHEAVYDB_ARCHIVE=""

function process_args(){
  while (( $# )); do
    case "$1" in
      --internally-built-pyheavydb)
        INTERNALLY_RELEASED_PYHEAVYDB=true
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

  mv ${PYHEAVYDB_ARCHIVE} ./dist
  PKG_PATH=./dist/${PYHEAVYDB_ARCHIVE}
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
