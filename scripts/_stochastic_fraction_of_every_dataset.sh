NEW_FILE_PATH=${WORK}/partial_bigscience_dataset/${1}
echo $NEW_FILE_PATH
mkdir -p "$(dirname "$NEW_FILE_PATH")"
perl -ne 'print if (rand() < 0.01)' ${1} > $NEW_FILE_PATH
