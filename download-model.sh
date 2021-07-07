# download pretrained model
mkdir -p models
wget https://www.dropbox.com/s/mjnclljcj160l97/bert-dg-special-tokens.zip?dl=1 -O bert-dg-special-tokens.zip
unzip bert-dg-special-tokens.zip -d models

# download our model
mkdir -p ckpt/DST/bert-dg-special-tokens_expand
wget https://www.dropbox.com/s/9dewh3kek7rzy6l/arguments.json?dl=1 -O ckpt/DST/bert-dg-special-tokens_expand/arguments.json
wget https://www.dropbox.com/s/of8mkht5ckw1ukl/checkpoint-6420.zip?dl=1 -O checkpoint-6420.zip
unzip checkpoint-6420.zip -d ckpt/DST/bert-dg-special-tokens_expand