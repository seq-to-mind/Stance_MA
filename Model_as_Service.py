import os
import sys
import time
import re
import pickle
import shutil

import json
import numpy as np
from urllib import parse
from flask import Flask, request, jsonify

import spacy
import nltk
import benepar
# benepar.download('benepar_en3')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

""" Loading LLM model """
machine_address = "127.0.0.1"
checkpoint_path = "../../pretrained_models/vicuna/vicuna-13b-v1.5/"
port_number = 2121
device = 'cuda:0'
print(machine_address, "Port:", port_number, "GPU:", device)

print("Loading tokenizer:", checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, legcy=False, use_fast=False)
print("Loading model checkpoint:", checkpoint_path)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.float16)
model = model.to(device).eval()

print("eos_token_id", model.config.eos_token_id)
generation_config = GenerationConfig(
    do_sample=False, eos_token_id=model.config.eos_token_id,
    num_beams=1, early_stopping=False,
)

""" Loading Constituency Model """
spacy.prefer_gpu()
spacy_model = spacy.load('en_core_web_trf')
spacy_model.add_pipe('benepar', config={'model': 'benepar_en3'})


def get_mentioned_entities(text):
    parsed_tree = nltk.Tree.fromstring(text)
    tmp_res = [" ".join(i.leaves()) for i in parsed_tree.subtrees() if i.label() == 'NP']
    tmp_NP_list = []
    for i in tmp_res:
        one_match = 0
        for j in tmp_res:
            if i in j:
                one_match += 1
        if one_match == 1:
            tmp_NP_list.append(i)

    tmp_NP_list = [i for i in tmp_NP_list if len(i) > 2 and i[0] != "#" and i.lower() not in ["them", "@user", "this", "that"]]
    return tmp_NP_list


print("Model and Flask interface is running...")
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        with torch.no_grad():
            prompt = request.form["llm_input"]
            LLM_gen_len = int(request.form["LLM_gen_len"])
            print("Flask service received >>> ", prompt[:20])
            if len(prompt) < 5:
                one_output = "<Error> Your input is too short!"
            else:
                one_input = tokenizer(prompt, return_tensors="pt", padding=False, return_token_type_ids=False).to(device)
                input_token_len = one_input.input_ids.size(1)

                one_output = model.generate(**one_input, generation_config=generation_config, max_new_tokens=LLM_gen_len)[0]
                one_output = tokenizer.decode(one_output[input_token_len:])
                one_output = "<Output> " + one_output.replace("\n", " ").strip()
                print(one_output[:20])
            return jsonify({'llm_output': one_output})


@app.route('/parse', methods=['POST'])
def parse():
    if request.method == 'POST':
        one_output = []
        with torch.no_grad():
            one_text = request.form["parse_input"]
            print("Flask service received >>> ", one_text[:20])
            if len(one_text) < 5:
                one_output = "<Error> Your input is too short!"
            else:
                parsed_doc = spacy_model(one_text)
                for one_sen in list(parsed_doc.sents):
                    one_output.extend(get_mentioned_entities(one_sen._.parse_string))
            return jsonify({'parse_output': one_output})


if __name__ == '__main__':
    app.run(host=machine_address, port=port_number)
