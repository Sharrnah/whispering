docker build builder/ -f builder/Dockerfile-linux64 -t whispering-tiger-builder:linux
docker run -v "$PWD:/src" whispering-tiger-builder:linux
