#!/usr/bin/env bash

# Basic test of example script functionality by running them alll
# Only supported under Linux.

SCRIPT=$(basename $0)
SCRIPTPATH=$(realpath $(dirname $0))
USAGE=$(cat <<-EOF
Usage: $SCRIPT [-h] [-d]
          [-h] Display usage information
          [-e] Display excerpt of error message on failure
EOF
)

OPTIND=1
DISPLAY_ERROR=0
while getopts ":he" opt; do
    case $opt in
    h) echo "$USAGE"; exit 0;;
    e) DISPLAY_ERROR=1;;
    \?) echo "Error: invalid option -$OPTARG" >&2
            echo "$USAGE" >&2
            exit 1
            ;;
    esac
done

shift $((OPTIND-1))
if [ ! $# -eq 0 ] ; then
    echo "Error: no positional arguments" >&2
    echo "$USAGE" >&2
    exit 2
fi

# Set environment variables and paths. This script is assumed to be run
# from its root directory.
export PYTHONPATH=$SCRIPTPATH/..
export PYTHONIOENCODING=utf-8
export MPLBACKEND=agg
export PYTHONWARNINGS=ignore:Matplotlib:UserWarning
d='/tmp/scriptcheck_'$$
mkdir -p $d
retval=0

# On SIGINT clean up temporary script directory and exit.
function cleanupexit {
    rm $d/*.py
    rmdir $d
    exit 2
}
trap cleanupexit SIGINT

# Define regex strings
re="s/input\(/#input\(/g; "

# Iterate over all scripts.
for f in $SCRIPTPATH/*.py; do
    printf "%-50s " $(basename $f)

    # Create temporary copy of script with all algorithm maxiter values set
    # to small number and final input statements commented out.
    g=$d/$(basename $f)
    sed -E -e "$re" $f > $g

    # Run temporary script and print status message.
    if output=$(timeout 60s python $g 2>&1); then
        printf "\033[0;32m%s\033[0m\n" succeeded
    else
        printf "\033[0;31m%s\033[0m\n" FAILED
        retval=1
    if [ $DISPLAY_ERROR -eq 1 ]; then
       echo "$output" | tail -8 | sed -e 's/^/    /'
    fi
    fi

    # Remove temporary script.
    rm -f $g

done

# Remove temporary script directory.
rmdir $d

exit $retval
