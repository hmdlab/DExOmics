#!/bin/bash

for gzfile in *.txt.gz; do
    folder="${gzfile%.txt.gz}"
    mkdir -p "../data/$folder"

    zcat "$gzfile" | while read url; do
        wget -P "../data/$folder" "$url"
    done

    for file in "../data/$folder"/*.gz; do
        if [[ -f "$file" ]]; then
            gunzip -f "$file"
        fi
    done
done


