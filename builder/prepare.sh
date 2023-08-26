#!/bin/bash

# Fail on errors.
set -e

# Make sure .bashrc is sourced
. /root/.bashrc

# Allow the workdir to be set using an env var.
# Useful for CI pipiles which use docker for their build steps
# and don't allow that much flexibility to mount volumes
WORKDIR=${SRCDIR:-/src}

# prepare nltk tokenizers before building using pyinstaller
mkdir -p ${WORKDIR}/.cache/
mkdir -p ${WORKDIR}/.cache/nltk/tokenizers/
wget -P "${WORKDIR}/.cache/nltk/tokenizers" https://s3.libs.space:9000/ai-models/nltk/tokenizers/punkt.zip
unzip -o ${WORKDIR}/.cache/nltk/tokenizers/punkt.zip -d ${WORKDIR}/.cache/nltk/tokenizers
rm ${WORKDIR}/.cache/nltk/tokenizers/punkt.zip
