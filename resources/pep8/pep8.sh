#!/usr/bin/env bash


autopep=$(autopep8 -d $1)
echo "$autopep" | colordiff

