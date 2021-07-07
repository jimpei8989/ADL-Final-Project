# Final Project - Dialogue State Tracking
> Applied Deep Learning (CSIE 5431)

## Shortcuts
- [Kaggle - Seen domains](https://www.kaggle.com/c/adl-final-dst-with-chit-chat-seen-domains/)
- [Kaggle - Unseen domains](https://www.kaggle.com/c/adl-final-dst-with-chit-chat-unseen-domains/)
- [Project SPEC slides](https://docs.google.com/presentation/d/1vekovUzNlffmbTyM4X3auGHt2P5PKUfDV2_eea5ycAU/view)
- [Project video](https://drive.google.com/file/d/1xiql4cxErLJonIzjV7XynHzobKLVjoTl/view)

## Submodules

> To use the submodules, `git submodule update --init --recursive`.

- [Trippy](https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public)
- [Dialoglue](https://github.com/alexa/dialoglue)

### How to run - NLG
Note: run `download-data.sh` first to get the dataset
#### Use pretrained model (BlenderBot)
```
python src/main_nlg.py --predict --test_data [test_data_path] --opt_file [opt_file_path]
```

#### Train on DST dataset
+ Train & predict begin chit-chat:
    ```
    python src/main_nlg.py --train_begin --pretrained t5-base --ckpt_dir [model_dir]
    python src/main_nlg.py --predict_begin --ckpt_dir [model_dir] --test_data [test_data_path] --opt_file [opt_file_path]
    ```
+ Train & predict end chit-chat:
    ```
    python src/main_nlg.py --train_end --pretrained t5-base --ckpt_dir [model_dir]
    python src/main_nlg.py --predict_end --ckpt_dir [model_dir] --test_data [test_data_path] --opt_file [opt_file_path]
    ```

#### Filtering
```
python src/postprocess.py -b [begin_chit-chat_file_path] -e [begin_chit-chat_file_path] -o [opt_json_path]
```

