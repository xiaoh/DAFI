
autopep=$(autopep8 -dr .)
echo "$autopep" | colordiff

