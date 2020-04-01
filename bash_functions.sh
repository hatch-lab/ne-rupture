#!/bin/bash
nerupture() {
  cd "${NE_RUPTURE_INSTALL_PATH}"
  source "${NE_RUPTURE_INSTALL_PATH}/.venv/bin/activate"
  python "${NE_RUPTURE_INSTALL_PATH}/preprocess.py" aics "${1}"
  if [ $? -ne 0 ]; then
    printf 'Preprocess failed'
  else
    python "${NE_RUPTURE_INSTALL_PATH}/classify.py" manual "${1}"
  fi
  deactivate
}