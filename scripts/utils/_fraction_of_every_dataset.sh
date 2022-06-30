NUM_LINES=$(wc -l < $1)
NUM_REMAINING_LINES=$((NUM_LINES/10000))
if [ "$NUM_REMAINING_LINES" -gt "0" ]; then
   NEW_FILE_PATH=${WORK}/partial_bigscience_dataset/${1}
   echo $NEW_FILE_PATH
   mkdir -p "$(dirname "$NEW_FILE_PATH")"
   head -$NUM_REMAINING_LINES $1 > $NEW_FILE_PATH
fi