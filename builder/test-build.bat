docker build builder/ -f builder/Dockerfile -t whispering-tiger-builder:win64
docker run -v "%cd%:/src" whispering-tiger-builder:win64
