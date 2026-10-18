#!/bin/bash
# Entrypoint for the Ascend NPU image: bring the CANN runtime onto the loader
# path, then exec locate-anything-cli with whatever args were passed.
set -e

# CANN toolkit runtime libs (libascendcl, libnnopbase, libopapi, ...).
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

# Ascend kernel-driver userspace libs (libascend_hal). The host driver tree is
# expected to be mounted read-only at /usr/local/Ascend/driver.
export LD_LIBRARY_PATH="${ASCEND_TOOLKIT_HOME}/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH}"

exec locate-anything-cli "$@"
