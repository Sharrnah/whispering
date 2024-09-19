#!/bin/bash -i

# Fail on errors.
set -e

# Make sure .bashrc is sourced
. /root/.bashrc

# Allow the workdir to be set using an env var.
# Useful for CI pipiles which use docker for their build steps
# and don't allow that much flexibility to mount volumes
WORKDIR=${SRCDIR:-/src}

#
# In case the user specified a custom URL for PYPI, then use
# that one, instead of the default one.
#
if [[ "$PYPI_URL" != "https://pypi.python.org/" ]] || \
   [[ "$PYPI_INDEX_URL" != "https://pypi.python.org/simple" ]]; then
    # the funky looking regexp just extracts the hostname, excluding port
    # to be used as a trusted-host.
    mkdir -p /root/.pip
    echo "[global]" > /root/.pip/pip.conf
    echo "index = $PYPI_URL" >> /root/.pip/pip.conf
    echo "index-url = $PYPI_INDEX_URL" >> /root/.pip/pip.conf
    echo "trusted-host = $(echo $PYPI_URL | perl -pe 's|^.*?://(.*?)(:.*?)?/.*$|$1|')" >> /root/.pip/pip.conf

    echo "Using custom pip.conf: "
    cat /root/.pip/pip.conf
fi

BUILD_DIST_DIR=${DIST_DIR:-./dist/linux}

cd $WORKDIR

echo "$@"

if [[ "$@" == "" ]]; then
    if [ -f requirements.txt ]; then
        # use --no-cache-dir to try to reduce memory usage. (see https://github.com/pypa/pip/issues/2984)
        pip install --no-cache-dir -r requirements-linux.txt -r requirements.nvidia.txt --no-build-isolation
    fi # [ -f requirements.txt ]

    pyinstaller --clean -y --dist ${BUILD_DIST_DIR} --workpath /tmp *.spec
    chown -R --reference=. ${BUILD_DIST_DIR}
else
    sh -c "$@"
fi # [[ "$@" == "" ]]
