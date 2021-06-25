if [[ -d dataset/ ]]; then
    echo "Error: directory \`dataset/\` already exists, remove it first."
    exit 1
fi

kaggle competitions download -c adl-final-dst-with-chit-chat-seen-domains \
    && unzip -q adl-final-dst-with-chit-chat-seen-domains.zip \
    && rm adl-final-dst-with-chit-chat-seen-domains.zip \
    && mkdir dataset \
    && mv seen_empty_submission.csv state_to_csv.py dataset/ \
    && mv data/data/ data-0610/data-0610/ data-0614/data-0614 data-0625/data-0625 dataset/ \
    && rm -rf data data-0610 data-0614 data-0625 \
