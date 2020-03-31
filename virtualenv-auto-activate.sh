#!/bin/bash
# virtualenv-auto-activate.sh
#
# Installatikon:
#    Add this line to your .bashrc or .bash_profile
#
# Adapted from gist.github.com/codysoyland/2198913
_virtualenv_auto_activate() {
  if [ -e ".venv" ]; then
    # Check to see if .venv is owned by this user
    if [ `stat -c "%u" .venv` = `id -u` ]; then
      # Check to see if already activated to avoid redundant activating
      if [ "$VIRTUAL_ENV" != "$(pwd -P)/.venv" ]; then
        _VENV_NAME=$(basename `pwd`)
        echo Activating virtualenv \"$_VENV_NAME"...
        VIRTUAL_ENV_DISABLE_PROMPT=1
        source .venv/bin/activate
        _OLD_VIRTUAL_PS1="$PS1"
        PS1="($_VENV_NAME)$PS1"
        export PS1
      fi
    fi
  fi
}

export PROMPT_COMMAND=_virtualenv_auto_activate