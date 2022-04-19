#!/usr/bin/env bash
# builds docker image used on this project.
docker build --network host -f ../Dockerfile -t salt-mask-vae:1.0 .

docker build --network host -f ../cpu.Dockerfile -t salt-mask-vae:cpu-1.0 .
