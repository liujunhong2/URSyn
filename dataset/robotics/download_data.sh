#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

wget http://downloads.cs.stanford.edu/juno/triangle_real_data.zip -O _tmp.zip

unzip _tmp.zip
rm _tmp.zip
