# ENV Requirements
## python3.10 (Installed)

INTERNALLY_RELEASED_PYHEAVYDB=false
HTTP_PYTHON_DEPS="https://dependencies.mapd.com/python-deps"

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

function test_for_pyheavydb_dev_requirements() {
  if [[ $INTERNALLY_RELEASED_PYHEAVYDB == "true" ]]; then
    ## Assume this happens after the dist dir has been
    ## made and the requirements.txt files has been copied
    if [[ ! -d ./dist ]] ; then 
      echo "WARNING. Either the dist dir has not been created or" \
        "the script is being run from the wrong place"
      return
    fi
    wget --continue ${HTTP_PYTHON_DEPS}/pyheavydb.whl.version
    PYHEAVYDB_ARCHIVE=$(head -n 1 pyheavydb.whl.version)
    wget --continue ${HTTP_PYTHON_DEPS}/${PYHEAVYDB_ARCHIVE}
    ## assume we're in the project root dir
    mv ${PYHEAVYDB_ARCHIVE} ./dist
    PKG_PATH=$(pwd)/dist/${PYHEAVYDB_ARCHIVE}
    sed -i "1 i ${PKG_PATH}" ./dist/requirements.txt
  fi
}

function test_for_pyheavydb_local_install() {
  if [[ $INTERNALLY_RELEASED_PYHEAVYDB == "true" ]]; then
    wget --continue ${HTTP_PYTHON_DEPS}/pyheavydb.whl.version
    PYHEAVYDB_ARCHIVE=$(head -n 1 pyheavydb.whl.version)
    wget --continue ${HTTP_PYTHON_DEPS}/${PYHEAVYDB_ARCHIVE}
    pip install ${PYHEAVYDB_ARCHIVE}
  fi
}

