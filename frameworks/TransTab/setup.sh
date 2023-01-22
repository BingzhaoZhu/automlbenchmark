#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/RyanWangZf/transtab.git"}
PKG=${3:-"scikit-learn"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="main"
fi

. "${HERE}/../shared/setup.sh" "${HERE}" true

PIP install --no-cache-dir -U git+https://github.com/RyanWangZf/transtab.git
