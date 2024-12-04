#!/bin/bash

mkdir -p data/raw

cd data/raw

# URLs to download
urls=(
  "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-fr.txt.zip"
  "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-es.txt.zip"
  "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-it.txt.zip"
  "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-pt.txt.zip"
  "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-ro.txt.zip"
)

# Download files in parallel
echo "Starting downloads, this may take a while..."
for url in "${urls[@]}"; do
  wget -q "$url" &
done

# Wait for all background downloads to complete
wait

echo "Downloads complete."

# Iterate through all zip files in the directory
for zip_file in ./*.zip; do
  # Extract the base filename (e.g., en-es.txt.zip -> en-es.txt)
  base_name=$(basename "$zip_file" .zip)

  # Extract the language pair (e.g., en-es)
  lang_pair=${base_name%.txt}

  # Unzip the file into the corresponding folder
  unzip "$zip_file" -d "$lang_pair"
done

echo "Extraction complete."

