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
XQuartz_package='https://github.com/XQuartz/XQuartz/releases/download/XQuartz-2.7.11/XQuartz-2.7.11.dmg'
XQuartz_package_name='XQuartz.pkg'

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
  if [ $# -gt 3 ]; then
    package_name="$4"
  fi
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

  extension="${filename##*.}"

  if [ "${extension}" = "dmg" ] && [ $# -gt 3 ]; then
    hdiutil attach "${filename}"
    volume="${filename%.*}"
    sudo installer -pkg "/Volumes/${volume}/${package_name}" -target /
    hdiutil detach "/Volumes/${volume}"
  else
    sudo installer -pkg "${filename}" -target /
  fi
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
brew install python@3.8 git wget cairo coreutils mysql

# Symlink JDK
if [ -d "/Library/Java/JavaVirtualMachines/openjdk-11.jdk" ]; then
  echo ""
else
  printf '
  Installing Java11. You may need to enter your password
  '
  sudo ln -sfn /usr/local/opt/openjdk@11/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-11.jdk
fi

echo "export JAVA_HOME=\"/Library/Java/JavaVirtualMachines/openjdk-11.jdk/Contents/Home/\"" >> ~/.bash_profile
echo "export PATH=\"\${JAVA_HOME}/bin:\$PATH\"" >> ~/.bash_profile
echo "export CPPFLAGS=\"-I/usr/local/opt/openjdk@11/include\"" >> ~/.bash_profile

echo "export JAVA_HOME=\"/Library/Java/JavaVirtualMachines/openjdk-11.jdk/Contents/Home/\"" >> ~/.zshrc
echo "export PATH=\"\${JAVA_HOME}/bin:\$PATH\"" >> ~/.zshrc
echo "export CPPFLAGS=\"-I/usr/local/opt/openjdk@11/include\"" >> ~/.zshrc

# Install R and XQuartz
install_package "R" ${R_package} ${R_sig}
install_package "XQuartz" ${XQuartz_package} ${XQuartz_sig} ${XQuartz_package_name}

# Clone repo
cd ~/Documents
printf '
'"${highlight_color}"'Enter the number corresponding to the folder you want to install this tool.'"${default_color}\n"
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

echo "export HATCH_LAB_NE_RUPTURE_TOOL_PATH=\"${HOME}/Documents/${d}ne-rupture\"" >> ~/.zshrc
echo "source \"\${HATCH_LAB_NE_RUPTURE_TOOL_PATH}/bash_functions.sh\"" >> ~/.zshrc
source ~/.zshrc

# Set up virtual env
/usr/local/opt/python@3.8/bin/python3 -m venv .venv
VIRTUAL_ENV="${HATCH_LAB_NE_RUPTURE_TOOL_PATH}/.venv"
PATH="${VIRTUAL_ENV}/bin:${PATH}"

pip install --upgrade pip
pip install -r requirements.txt
printf '
'"${highlight_color}"'Finished!'"${default_color}"'
'