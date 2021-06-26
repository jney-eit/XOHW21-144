#!/bin/bash

mkdir -p build
cd ./build

if [ -e ./main ]
then
	sudo ./main
else
	rm -r *
	cmake ..
	make
	sudo ./main
fi
