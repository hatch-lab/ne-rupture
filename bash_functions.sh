ner() {
  if [ $# -lt 2 ]; then
    printf '
Usage:
ner segment [folder]
  Segment the TIFFs located in [folder]/images/raw

ner annotate [folder]
  Annotate the segmented TIFFs located in [folder]
'
    return
  fi

  source "${HATCH_LAB_NE_RUPTURE_TOOL_PATH}/.venv/bin/activate"
  abs_path="`realpath \"${2}\"`"

  if [ "${1}" = "segment" ]; then
    python "${HATCH_LAB_NE_RUPTURE_TOOL_PATH}/preprocess.py" aics "${abs_path}" "${@:3}"
  elif [ "${1}" = "annotate" ]; then
    python "${HATCH_LAB_NE_RUPTURE_TOOL_PATH}/classify.py" manual "${abs_path}" "${@:3}"
  else
    echo "You may specify ner segment or ner annotate"
  fi
  deactivate
}
