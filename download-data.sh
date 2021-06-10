kaggle competitions download -c adl-final-dst-with-chit-chat-seen-domains \
    && unzip -q adl-final-dst-with-chit-chat-seen-domains.zip \
    && rm adl-final-dst-with-chit-chat-seen-domains.zip \
    && mv data/ dataset/ \
    && mv data-0610/data-0610/ dataset/ \
    && rm -rf data-0610 \
    && mv seen_empty_submission.csv state_to_csv.py dataset/ \
    && rm -rf data
