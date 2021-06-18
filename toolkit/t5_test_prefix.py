from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

import torch
import os

device = torch.device("cuda:0")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


pretrained_name = "t5-base"
tok = AutoTokenizer.from_pretrained(pretrained_name)
model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_name)
model.to(device)

texts = [
    "the weather that day will be about 93 degrees with a 0 percent chance of rain.",
    "it's been booked and you'll receive a confirmation email shortly. you can reach them at 925-930-7450.",
    "Please confirm to play Adorn on the TV.",
]

while True:
    print("Input prefix:", end=" ")
    prefix = input().rstrip()
    if prefix[-1] != ":":
        prefix += ":"

    for t in texts:
        inp_before = prefix + " " + t
        inp = torch.LongTensor(tok.encode(inp_before)).unsqueeze(0)
        with torch.no_grad():
            ret = model.generate(inp.to(device), num_beams=10, max_length=64)[0]
            ret = tok.decode(ret, skip_special_tokens=True)
            print(f"input: {inp_before}")
            print(f"output: {ret}")
    print("----------")
