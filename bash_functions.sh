#!/bin/bash
nerupture() {
  cd "${HATCH_LAB_NE_RUPTURE_TOOL_PATH}"
  source "${HATCH_LAB_NE_RUPTURE_TOOL_PATH}/.venv/bin/activate"
  python "${HATCH_LAB_NE_RUPTURE_TOOL_PATH}/preprocess.py" aics "${1}"
  if [ $? -ne 0 ]; then
    printf 'Preprocess failed'
  else
    python "${HATCH_LAB_NE_RUPTURE_TOOL_PATH}/classify.py" manual "${1}"
  fi
  deactivate
}