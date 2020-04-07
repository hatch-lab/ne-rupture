ner() {
  source "${HATCH_LAB_NE_RUPTURE_TOOL_PATH}/.venv/bin/activate"
  if [ "${1}" -eq "segment" ]; then
    python "${HATCH_LAB_NE_RUPTURE_TOOL_PATH}/preprocess.py" aics "${@:2}"
  elif [ "${1}" -eq "annotate" ]; then
    python "${HATCH_LAB_NE_RUPTURE_TOOL_PATH}/classify.py" manual "${@:2}"
  else
    echo "You may specify ner segment or ner annotate"
  fi
  deactivate
}