#!/bin/bash


#Check if argument is directory
if ! [[ -d "$1" ]]; then
	echo "MP3 directory $1 is not a valid directory."
	exit 1
fi

if ! [[ -d "$2" ]]; then
	echo "Saving WAV directory $2 is not a valid directory..."
	exit 1
fi	

FILE_DIR=$1
SAVE_DIR=$2

# Remove trailing slash
FILE_DIR=${FILE_DIR%/}
SAVE_DIR=${SAVE_DIR%/}

echo $FILE_DIR
echo $SAVE_DIR

for filepath in $FILE_DIR/*; do
	if [ ${filepath: -4} == ".mp3" ]; then
		#echo Converting $filepath
		filename="$(basename "$filepath")"
		# remove extension
		filename="${filename%.*}"
		# Convert mp3 to wav
		mpg123 -w "$SAVE_DIR"/"$filename".wav "$filepath"
	fi
done
