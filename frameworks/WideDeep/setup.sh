#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/jrzaurin/pytorch-widedeep.git"}
PKG=${3:-"pytorch-widedeep"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

# creating local venv
. ${HERE}/../shared/setup.sh ${HERE} true

PIP install --upgrade pip
PIP install --upgrade setuptools wheel

if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir -U "${PKG}"
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U "${PKG}==${VERSION}"
fi

PY -c "from pytorch-widedeep.version import __version__; print(__version__)" >> "${HERE}/.setup/installed"
