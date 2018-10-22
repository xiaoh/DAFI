#!/usr/bin/env bash

autopep=$(autopep8 -dr .)
echo "$autopep" | colordiff

