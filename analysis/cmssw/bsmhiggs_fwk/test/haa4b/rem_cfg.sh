#!/usr/bin/env bash

for file in *.root; do
    py_file="${file%.root}_cfg.py"
    if [ -f "$py_file" ]; then
        rm "$py_file"
    fi
done
