ner() {
  source "${HATCH_LAB_NE_RUPTURE_TOOL_PATH}/.venv/bin/activate"
  if [ abs_path="`dir_resolve \"${2}\"`" ]; then
    # Nothing
  else
    echo "The provided path ${2} does not exist"
  fi
  if [ "${1}" = "segment" ]; then
    python "${HATCH_LAB_NE_RUPTURE_TOOL_PATH}/preprocess.py" aics "${abs_path}" "${@:3}"
  elif [ "${1}" = "annotate" ]; then
    python "${HATCH_LAB_NE_RUPTURE_TOOL_PATH}/classify.py" manual "${abs_path}" "${@:3}"
  else
    echo "You may specify ner segment or ner annotate"
  fi
  deactivate
}

# From: https://stackoverflow.com/a/7126780
dir_resolve() {
  # cd to the directory; silence any errors, but return the error code
  cd "$1" 2>/dev/null || return $?
  echo "`pwd -P`"
}