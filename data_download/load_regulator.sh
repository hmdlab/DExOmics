#!/bin/bash

for txtfile in *.txt; do
    folder="${txtfile%.txt}"
    mkdir -p "../data/$folder"

    while read url; do
        wget -P "../data/$folder" "$url"
    done < "$txtfile"

    for gzfile in "../data/$folder"/*.gz; do
        if [[ -f "$gzfile" ]]; then
            gunzip -f "$gzfile"
        fi
    done
done


