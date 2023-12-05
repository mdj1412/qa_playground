import os
import argparse
import json
import pickle as pkl
import logging

import numpy as np
import torch

from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from model_utils import convert_model_to_int8_on_gpu
from eval_utils import f1_score, exact_match_score

with open("hf_access_token.txt") as f:
    access_token = f.read().strip()

def load_triviaqa(split, n_samples=None):
    assert split in ["train", "validation", "test"]
    dataset = load_dataset("trivia_qa", "rc.nocontext", split=split)
    dataset = [{"question": dp["question"], "answer": dp["answer"]} for dp in dataset]
    print (f"Loaded {len(dataset)} examples from {split}")

    if n_samples is not None:
        np.random.seed(2023)
        indices = np.random.permutation(range(len(dataset)))[:n_samples]
        dataset = [dataset[idx] for idx in indices]

    return dataset

def load_tokenizer_and_model(model_name):
    if model_name.startswith("llama"):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=access_token)
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=access_token)
    elif model_name.startswith("pythia"):
        tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/{model_name}-deduped")
        model = AutoModelForCausalLM.from_pretrained(f"EleutherAI/{model_name}-deduped")
    else:
        raise NotImplementedError()

    model = convert_model_to_int8_on_gpu(model, device="cuda")
    return tokenizer, model

def main(logger, args):
    tokenizer, model = load_tokenizer_and_model(args.model_name)

    question = "Who is the president of the United States?"
    answer = "Joe Biden"

    input_text = f"Question: {question}\nAnswer:"
    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]
    input_ids = input_ids.cuda()

    outputs = model.generate(input_ids=input_ids, num_beams=5, do_sample=True, max_new_tokens=10)
    generation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assert generation.startswith(input_text)
    prediction = generation[len(input_text):].split("\n")[0].strip()
    logger.info(prediction)
    logger.info(exact_match_score(prediction, answer))
    logger.info(f1_score(prediction, answer))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        default="llama-7b",
                        choices=["llama-7b", "pythia-160m", "pythia-410m", "pythia-1.4b", "pythia-2.8b", "pythia-6.9b"])
    parser.add_argument('--n_samples',
                        type=int,
                        default=None)
    parser.add_argument('--log_file',
                        type=str,
                        default=None)
    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    if args.log_file is not None:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(logger, args)


