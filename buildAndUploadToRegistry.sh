#!/usr/bin/env bash

# build image
IMAGE=registry.datexis.com/s91000/thesis-tuning

version=0.0.12
echo "Version: $version"
#docker login -u $1 -p $2 registry.datexis.com
docker build -t $IMAGE -t $IMAGE:$version .
docker push $IMAGE:$version
echo "Done pushing image $image for build $version"