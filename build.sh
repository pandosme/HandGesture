docker build --progress=plain --no-cache --pull=false --build-arg CHIP=aarch64 . -t detectx
docker cp $(docker create detectx):/opt/app ./build
cp ./build/*.eap .
rm -rf ./build
