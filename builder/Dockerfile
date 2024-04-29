FROM ubuntu:22.04

ENV DEBIAN_FRONTEND noninteractive

ARG WINE_VERSION=winehq-stable
ARG PYTHON_VERSION=3.11.6
ARG PYINSTALLER_VERSION=6.6

# we need wine for this all to work, so we'll use the PPA
RUN set -x \
    && dpkg --add-architecture i386 \
    && apt-get update -qy \
    && apt-get install --no-install-recommends -qfy gpg-agent rename apt-transport-https software-properties-common git winbind cabextract wget curl zip unzip xvfb xdotool x11-utils xterm p7zip-full wget \
    && wget -nv https://dl.winehq.org/wine-builds/winehq.key \
    && apt-key add winehq.key \
    && add-apt-repository 'https://dl.winehq.org/wine-builds/ubuntu/' \
    && apt-get update -qy \
    && apt-get install --no-install-recommends -qfy $WINE_VERSION \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && wget -nv https://raw.githubusercontent.com/Winetricks/winetricks/master/src/winetricks \
    && chmod +x winetricks \
    && mv winetricks /usr/local/bin

# wine-gecko
RUN mkdir -p /usr/share/wine/gecko
RUN curl -o /usr/share/wine/gecko/wine_gecko-2.47-x86.msi http://dl.winehq.org/wine/wine-gecko/2.47/wine_gecko-2.47-x86.msi
RUN curl -o /usr/share/wine/gecko/wine_gecko-2.47-x86_64.msi http://dl.winehq.org/wine/wine-gecko/2.47/wine_gecko-2.47-x86_64.msi

# wine settings
ENV WINEARCH win64
ENV WINEDEBUG fixme-all
ENV WINEPREFIX /wine
ENV WINEPATH 'C:\git\bin'

# xvfb settings
# from https://github.com/engineervix/docker-pyinstaller/blob/b99163b6eb0a81fc9a0bfd498c09c4e0253d01b2/Dockerfile-py3-win64#L56-L59
ENV DISPLAY :0
RUN set -x \
    && echo 'Xvfb $DISPLAY -screen 0 1024x768x24 &' >> /root/.bashrc


# PYPI repository location
ENV PYPI_URL=https://pypi.python.org/
# PYPI index location
ENV PYPI_INDEX_URL=https://pypi.python.org/simple

# install python in wine, using the msi packages to install, extracting
# the files directly, since installing isn't running correctly.
RUN set -x \
    && winetricks win10 \
    && for msifile in `echo core dev exe lib path pip tcltk tools`; do \
        wget -nv "https://www.python.org/ftp/python/$PYTHON_VERSION/amd64/${msifile}.msi"; \
        wine msiexec /i "${msifile}.msi" /qb TARGETDIR=C:/Python310; \
        rm ${msifile}.msi; \
    done \
    && cd /wine/drive_c/Python310 \
    && echo 'wine '\''C:\Python310\python.exe'\'' "$@"' > /usr/bin/python \
    && echo 'wine '\''C:\Python310\Scripts\easy_install.exe'\'' "$@"' > /usr/bin/easy_install \
    && echo 'wine '\''C:\Python310\Scripts\pip.exe'\'' "$@"' > /usr/bin/pip \
    && echo 'wine '\''C:\Python310\Scripts\pyinstaller.exe'\'' "$@"' > /usr/bin/pyinstaller \
    && echo 'wine '\''C:\Python310\Scripts\pyupdater.exe'\'' "$@"' > /usr/bin/pyupdater \
    && echo 'assoc .py=PythonScript' | wine cmd \
    && echo 'ftype PythonScript=c:\Python310\python.exe "%1" %*' | wine cmd \
    && while pgrep wineserver >/dev/null; do echo "Waiting for wineserver"; sleep 1; done \
    && chmod +x /usr/bin/python /usr/bin/easy_install /usr/bin/pip /usr/bin/pyinstaller /usr/bin/pyupdater \
    && (pip install -U pip || true) \
    && rm -rf /tmp/.wine-*

# upgrade pip
RUN wine cmd /C "python.exe -m pip install --upgrade pip"

ENV W_DRIVE_C=/wine/drive_c
ENV W_WINDIR_UNIX="$W_DRIVE_C/windows"
ENV W_SYSTEM64_DLLS="$W_WINDIR_UNIX/system32"
ENV W_TMP="$W_DRIVE_C/windows/temp/_$0"

# install Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019 dll files
RUN set -x \
    && rm -f "$W_TMP"/* \
    && wget -P "$W_TMP" https://aka.ms/vs/16/release/vc_redist.x64.exe \
    && cabextract -q --directory="$W_TMP" "$W_TMP"/vc_redist.x64.exe \
    && cabextract -q --directory="$W_TMP" "$W_TMP/a11" \
    && cabextract -q --directory="$W_TMP" "$W_TMP/a12" \
    && cabextract -q --directory="$W_TMP" "$W_TMP/a13" \
    && cd "$W_TMP" \
    && rename 's/_/\-/g' *.dll \
    && cp "$W_TMP"/*.dll "$W_SYSTEM64_DLLS"/

# install git for windows
RUN set -x \
    && mkdir -p /wine/drive_c/git \
    && wget -P "/wine/drive_c/git/" https://github.com/git-for-windows/git/releases/download/v2.42.0.windows.1/PortableGit-2.42.0-64-bit.7z.exe \
    && 7za x /wine/drive_c/git/PortableGit-2.42.0-64-bit.7z.exe -o/wine/drive_c/git/ \
    && rm -rf /wine/drive_c/git/PortableGit-2.42.0-64-bit.7z.exe \
    && echo 'wine '\''C:\git\bin\git.exe'\'' "$@"' > /usr/bin/git \
    && while pgrep wineserver >/dev/null; do echo "Waiting for wineserver"; sleep 1; done \
    && chmod +x /wine/drive_c/git/bin/git.exe /usr/bin/git \
    && wine cmd /C "echo %PATH%" \
    && wine cmd /C "git -v" \

ENV SRCDIR=${SRCDIR:-/src}
ENV DIST_DIR=${DIST_DIR:-./dist/windows}

# install pyinstaller
RUN /usr/bin/pip install wheel pyinstaller==$PYINSTALLER_VERSION

# install ninja builder
#RUN /usr/bin/pip install ninja

# put the src folder inside wine
RUN mkdir /src/ && ln -s /src /wine/drive_c/src
VOLUME /src/
WORKDIR /wine/drive_c/src/
RUN mkdir -p /wine/drive_c/tmp

COPY entrypoint-windows.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
