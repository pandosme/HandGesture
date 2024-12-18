docker build  --progress=plain --no-cache  --build-arg CHIP=artpec7 . -t detectx
docker cp $(docker create detectx):/opt/app ./build
cp ./build/*.eap .
rm -rf ./build
