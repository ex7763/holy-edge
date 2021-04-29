#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Wrong parameters! Need one."
    exit 1;
fi

(cd $1 && 
    find . -type f -name 'model*' -exec rm {} + &&
    find . -type f -name 'checkpoint' -exec rm {} + &&

    # Useless model path
    find . -type d -name '*lr*penalty*' -exec rm -rf {} + 
)
