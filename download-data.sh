kaggle competitions download -c adl-final-dst-with-chit-chat-seen-domains \
    && unzip -q adl-final-dst-with-chit-chat-seen-domains.zip \
    && rm adl-final-dst-with-chit-chat-seen-domains.zip \
    && mv data/data/ dataset/ \
    && mv seen_empty_submission.csv state_to_csv.py dataset/ \
    && rm -rf data
