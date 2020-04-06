#!/bin/bash

# Adapted from macos-guest-virtualbox.sh
# URL: https://github.com/myspaghetti/macos-guest-virtualbox
# GPL 3.0

# terminal text colors
warning_color="\e[48;2;255;0;0m\e[38;2;255;255;255m" # white on red
highlight_color="\e[48;2;0;0;9m\e[38;2;255;255;255m" # white on black
low_contrast_color="\e[48;2;0;0;9m\e[38;2;128;128;128m" # grey on black
default_color="\033[0m"

R_sig='2677aaf9da03e101f9e651c80dbec25461479f56'
R_package='http://ftp.ussg.iu.edu/CRAN/bin/macosx/R-3.6.3.pkg'

XQuartz_sig='787b238fb09fec56665f2badf896a2e83e4fe2e0'
XQuartz_package='https://dl.bintray.com/xquartz/downloads/XQuartz-2.7.11.dmg'

branch="stable"
if [ -f ".dev" ]; then
  branch="dev"
fi

function clear_input_buffer_then_read() {
  while read -d '' -r -t 0; do read -d '' -t 0.1 -n 10000; break; done
  read
}

function install_package() {
  name="$1"
  url="$2"
  signature="$3"
  printf '
Installing '"${name}"'
'"${highlight_color}"'You may need to enter your password'"${default_color}"'
'
  wget --quiet --continue --show-progress ${url}
  filename="${url##*/}"
  checksum=`shasum ${filename}`

  if [ "${checksum}" != "${signature}  ${filename}" ]; then
    printf '
    '"${warning_color}${name}"' checksum does not match. Exiting.'"${default_color}"'
    '
    exit 1
  fi

  sudo installer -pkg ${filename} -target /
  rm -f ${filename}
}

set -u

printf '
=======       Hatch Lab Nuclear Envelope Rupture analysis tool       =======

This script installs NE rupture tool along with any dependencies.

'"${highlight_color}"'Press enter to continue, CTRL-C to quit'"${default_color}"
clear_input_buffer_then_read

# Install Homebrew
printf '
Installing Homebrew package manager
'
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"

# Install brew packages
printf '
Installing software packages
'
brew install python3 git wget

# Install R and XQuartz
install_package "R" ${R_package} ${R_sig}
install_package "XQuartz" ${XQuartz_package} ${XQuartz_sig}

# Clone repo
cd ~/Documents
printf '
'"${highlight_color}"'Where do you want to install the tool?'"${default_color}\n"
select d in */; do test -n "${d}" && break; exit 1; done

printf '
Installing NE rupture tool to ~/Documents/'"${d}"' with branch '"${highlight_color}${branch}${default_color}"'
'
cd "${d}"
if [ -d "ne-rupture" ]; then
  cd ne-rupture
  git pull origin ${branch}
else
  git clone --recurse-submodules --branch ${branch} https://github.com/hatch-lab/ne-rupture.git
  cd ne-rupture
fi

echo "export HATCH_LAB_NE_RUPTURE_TOOL_PATH=\"${HOME}/Documents/${d}ne-rupture\"" >> ~/.bash_profile
echo "source \"\${HATCH_LAB_NE_RUPTURE_TOOL_PATH}/bash_functions.sh\"" >> ~/.bash_profile
source ~/.bash_profile

# Set up virtual env
python3 -m venv .venv
source .venv/bin/activate & process_id=$!
wait $process_id
if [ $? -ne 0 ]; then
  printf '
    '"${warning_color}"'Unable to activate virtual environment.'"${default_color}"'
    '
    exit 1
fi
pip install --upgrade pip
pip install -r requirements.txt
deactivate
printf '
'"${highlight_color}"'Finished!'"${default_color}"
''