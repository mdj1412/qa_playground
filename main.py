import os
import argparse
import json
import pickle as pkl
import logging
import time
import traceback
import psutil
import itertools

import numpy as np
import torch

from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from model_utils import convert_model_to_int8_on_gpu
from eval_utils import f1_score, exact_match_score
import select_demo
from select_demo import ordering

pid = os.getpid()
python_process = psutil.Process(pid)

def print_mem_use():
    memUse = python_process.memory_info()[0]/2.**30 
    print('memory use: %.1fGB' % memUse)

with open("hf_access_token.txt") as f:
    access_token = f.read().strip()

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print (f"device: {device}")

def load_triviaqa(split, n, demo_seed=2023):
    assert split in ["train", "validation", "test"]
    dataset = load_dataset("trivia_qa", "rc.nocontext", split=split)
    dataset = [{"question": dp["question"], "answer": dp["answer"]["value"]} for dp in dataset] if split=="train" \
        else [{"question": dp["question"], "answer": dp["answer"]} for dp in dataset]
    print (f"Loaded {len(dataset)} examples from {split}")

    # Randomness about demo_seed
    if split == "train": 
        np.random.seed(demo_seed)
    elif split == "validation":
        np.random.seed(2023)
    else: 
        raise NotImplementedError()
    indices = np.random.permutation(range(len(dataset)))[:n]

    dataset = [dataset[idx] for idx in indices]# (k, 1)
    print (f"Shape of dataset {np.array(dataset).shape}")
    return dataset

def load_nqopen(split, n, demo_seed=2023):
    assert split in ["train", "validation"]
    dataset = load_dataset("nq_open", split=split)
    dataset = [{"question": dp["question"], "answer": dp["answer"]} for dp in dataset]
    print (f"Loaded {len(dataset)} examples from {split}")

    # Randomness about demo_seed
    if split == "train": 
        np.random.seed(demo_seed)
    elif split == "validation": 
        np.random.seed(2023)
    else: 
        raise NotImplementedError()
    indices = np.random.permutation(range(len(dataset)))[:n]

    dataset = [dataset[idx] for idx in indices]# (k, 1)
    print (f"Shape of return dataset {np.array(dataset).shape}")
    return dataset

def load_train_and_test(k, n_test_samples, demo_seed, retrieval_based_prompt_selection, n_retrieval_data):
    train_n = n_retrieval_data if retrieval_based_prompt_selection else k
    print (f"Load Dataset - {train_n} train data and {n_test_samples} test data"); start = time.perf_counter()
    
    if args.dataset == "trivia_qa":
        train_dataset = load_triviaqa("train", n=train_n, demo_seed=demo_seed)
        test_dataset = load_triviaqa("validation", n=n_test_samples, demo_seed=2023)# Fixed demo_seed, always evaluate same test data
    elif args.dataset == "nq_open":
        train_dataset = load_nqopen("train", n=train_n, demo_seed=demo_seed)
        test_dataset = load_nqopen("validation", n=n_test_samples, demo_seed=2023)# Fixed demo_seed, always evaluate same test data
    else:
        raise NotImplementedError()

    logger.info(f"Finish load dataset (train,test) : {time.perf_counter()-start} sec")
    return train_dataset, test_dataset

def load_tokenizer_and_model(model_name):
    print (f"Load {model_name} model and tokenizer"); start = time.perf_counter()
    if model_name.startswith("llama"):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=access_token)
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=access_token)
    elif model_name.startswith("pythia"):
        tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/{model_name}-deduped")
        model = AutoModelForCausalLM.from_pretrained(f"EleutherAI/{model_name}-deduped")
    else:
        raise NotImplementedError()

    model = convert_model_to_int8_on_gpu(model, device)# Use GPU Memory
    logger.info(f"Finish load {model_name} model and tokenizer : {time.perf_counter()-start} sec")
    return tokenizer, model

def prompting(dataset, indices):
    if len(indices) == 0: return ""

    samples = "\n\n".join(f"Question: {dataset[idx]['question']}\nAnswer: {dataset[idx]['answer']}" \
                                for idx in indices)
    return samples + "\n\n"

def main(logger, args):
    print (f"Start demo_seed : {args.demo_seed}")
    tokenizer, model = load_tokenizer_and_model(args.model_name)# Use GPU Memory
    train_dataset, test_dataset = load_train_and_test(args.k, args.n_test_samples, args.demo_seed, \
                                            args.retrieval_based_prompt_selection, args.n_retrieval_data)
    assert len(test_dataset)==args.n_test_samples

    if (args.retrieval_based_prompt_selection) or \
        ((not args.retrieval_based_prompt_selection) and (args.ordering != None)):
        nns = select_demo.NearestNeighborSelector(
            args.sentence_encoder, 
            train_dataset, 
            metric=args.retrieval_metric, 
            logger=logger,
            device='cuda'
        )
        train_dataset_indicing_about_test_data = nns.select_k(
            args.k, test_dataset, demo_seed=args.demo_seed, dataset_name=args.dataset
        )

    em_list, f1_list = [], []; start = time.perf_counter()
    for idx, test_data in enumerate(tqdm(test_dataset)):
        answer = test_data["answer"]

        if args.dataset == "trivia_qa":
            possible_answer = [
                value for key, value_list in answer.items() 
                if key in ['aliases', 'normalized_aliases', 'normalized_value', 'value'] 
                for value in ([value_list] if type(value_list) == str else value_list)
            ]
        elif args.dataset == "nq_open":
            possible_answer = answer
        else:
            raise NotImplementedError()

        # Set of training examples and Ordering
        if (args.retrieval_based_prompt_selection) or \
        ((not args.retrieval_based_prompt_selection) and (args.ordering != None)):
            sample_indices = train_dataset_indicing_about_test_data[idx]# Select demonstrations
            sample_indices = ordering(sample_indices, args.permu_seed, args.ordering)# Permutation
        else:
            sample_indices = ordering(range(len(train_dataset)), args.permu_seed, args.ordering)# Permutation

        # Consider about all_permutations_majority_voting or all_permutations_mean_embedding
        if args.ordering != None and args.ordering.startswith("all_permutations"):
            all_permutations = list(itertools.permutations(sample_indices))
            num_loop = len(all_permutations)
        elif args.k == 0:# 0-shot
            all_permutations = [[]]
            num_loop = 1
        else :# Random(None) / low_to_high / high_to_low
            all_permutations = [ sample_indices ]
            num_loop = 1

        all_permutation_outputs = []
        for n_permu, sample_indices in enumerate(all_permutations):
            print (f"sample_indices : {sample_indices}")
            # Prompt Format
            samples = prompting(train_dataset, sample_indices)
            input_text = f"{samples}Question: {test_data['question']}\nAnswer:"

            input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]
            input_ids = input_ids.cuda()

            try:
                outputs = model.generate(input_ids=input_ids, num_beams=5, do_sample=True, max_new_tokens=10)
                generation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            except Exception as e:
                logger.info(e)
                logger.info(f"Number of token: {input_ids.size()}")

            #if not generation.startswith(input_text): from IPython import embed; embed()
            #assert generation.startswith(input_text)
            output = generation[len(input_text):].split("\n")[0].strip()
            all_permutation_outputs.append(output)# shape: (k!) or (1)

        # Consider about all_permutations_majority_voting or all_permutations_mean_embedding
        if args.ordering != None and args.ordering.startswith("all_permutations"):
            if args.ordering == "all_permutations_majority_voting":
                prediction = nns.majority_voting_prediction(all_permutation_outputs)
            elif args.ordering == "all_permutations_mean_embedding":
                prediction = nns.nearest_prediction(all_permutation_outputs)
            else:
                raise NotImplementedError()
        else :# Random(None) / low_to_high / high_to_low
            prediction = all_permutation_outputs[0]

        em = exact_match_score(prediction, possible_answer)
        f1 = f1_score(prediction, possible_answer)

        if idx%(args.n_test_samples//5)==0:# Check about 5 times for debugging purposes.
            logger.info(f"{idx+1} / {args.n_test_samples}\n{input_text}")
            logger.info(prediction)
            logger.info(em)# EM: exactly match
            logger.info(f1)# F1: similiarity same
        if idx%10==0: print (f"Accuracy - {idx} - EM: {em}  F1: {f1}")

        em_list.append(em); f1_list.append(f1)

    print ("Accuracy (Metric)")
    em = em_list.count(True)/len(em_list); f1 = np.mean(f1_list)
    logger.info(f"demo_seed: {args.demo_seed} - Avg EM: {em}  Avg F1: {f1})")
    logger.info(f"Finish Generating All Questions : {time.perf_counter()-start} sec")

    return em, f1

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        default="llama-7b",
                        choices=["llama-7b", "pythia-160m", "pythia-410m", "pythia-1.4b", "pythia-2.8b", "pythia-6.9b"])
    parser.add_argument('--n_test_samples',
                        type=int,
                        default=500,
                        help='Number of Test Samples')
    parser.add_argument('--k',
                        type=int,
                        default=0,
                        help='Number of demonstrations')
    parser.add_argument('--log_file',
                        type=str,
                        default=None,
                        help='Record the logger.info')
    parser.add_argument('--dataset',
                        type=str,
                        default="trivia_qa",
                        choices=["trivia_qa", "nq_open"],
                        help='Choose Dataset')
    parser.add_argument('--steps', '-s', type=str, default=None, help='Writing''steps accuracy when validation or testing')
    parser.add_argument('--demo_seed',
                        type=int,
                        default=2023,
                        help='Random Seeds (Randomness). A set of demonstrations (selection)')
    parser.add_argument('--demo_seeds',
                        type=str,
                        default=None,
                        help='A set of demo_seeds. For example, --demo_seed=2018,2019,2020 or --demo_seed=2023 \
                            [ Random Seeds (Randomness) : A set of demonstrations (selection). ]')
    parser.add_argument('--permu_seed',
                        type=int,
                        default=2023,
                        help='Random Seeds (Randomness). A permutation (ordering) of the training examples.')
    parser.add_argument('--retrieval_based_prompt_selection',
                        action="store_true",
                        help='Used retrieval based prompt selection or random sampling (if false)')
    parser.add_argument('--ordering',
                        type=str,
                        default=None,
                        choices=[None, "low_to_high", "high_to_low", "all_permutations_majority_voting", "all_permutations_mean_embedding"],
                        help='Random Seeds (Randomness). Choose low_to_high, high_to_low, all_permutations_majority_voting, all_permutations_mean_embedding')
    parser.add_argument('--sentence_encoder',
                        type=str,
                        default="princeton-nlp/sup-simcse-roberta-large",
                        choices=["princeton-nlp/unsup-simcse-bert-large-uncased", "princeton-nlp/unsup-simcse-roberta-large",
                                "princeton-nlp/sup-simcse-bert-large-uncased", "princeton-nlp/sup-simcse-roberta-large"],
                        help='Choose princeton-nlp/unsup-simcse-bert-large-uncased, princeton-nlp/unsup-simcse-roberta-large, \
                            princeton-nlp/sup-simcse-bert-large-uncased, princeton-nlp/sup-simcse-roberta-large')
    parser.add_argument('--n_retrieval_data',
                        type=int,
                        default=None,
                        help='Number of retrieval data')
    parser.add_argument('--retrieval_metric',
                        type=str,
                        default="euclidean",
                        choices=["euclidean", "cosine"],
                        help='NearestNeighborSelector with k, euclidean or cosine')
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


    #print ("Mean of Accuracy / Variance of Accuracy")
    #logger.info(f" Variance (em): {np.var(em_list)}  Variance (f1): {np.var(f1_list)}")
    
    try:
        start = time.perf_counter()

        if args.demo_seeds != None:# Computing "Mean of Accuracy" and "Variance of Accuracy"
            demo_seeds = list( map(int, args.demo_seeds.split(",")) )

            em_list = []; f1_list = []
            for demo_seed in demo_seeds:
                args.demo_seed = demo_seed; start2 = time.perf_counter()

                em, f1 = main(logger, args)
                em_list.append(em); f1_list.append(f1)

                logger.info(f"Finish main.py ( One of {args.demo_seeds} : {demo_seed} ) : {time.perf_counter()-start2} sec")
            logger.info(f"[ Mean of Accuracy (EM) , Variance of Accuracy (EM) ] : {np.mean(em_list)} , {np.var(em_list)}")
            logger.info(f"[ Mean of Accuracy (F1) , Variance of Accuracy (F1) ] : {np.mean(f1_list)} , {np.var(f1_list)}")
        
        else :# Computing only one demo_seed
            main(logger, args)

        logger.info(f"Finish the Command : {time.perf_counter()-start} sec")
    except Exception as e:
        logger.info(e)
        logging.error(traceback.format_exc())