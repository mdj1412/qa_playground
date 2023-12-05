import torch
import numpy as np

from tqdm import tqdm
import time
import os
from collections import Counter

from sklearn.metrics import pairwise
from sklearn.neighbors import NearestNeighbors
import pickle
from scipy.spatial import distance

from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader

print ("Load select_demo.py !")

class NearestNeighborSelector():
    """
        Retrieval-based prompt selection approach

        [ Sentence Embeddings for Retrieval ]

        Used model (Sentence Encoder, retrieval module):
        1. princeton-nlp/unsup-simcse-bert-large-uncased
        2. princeton-nlp/unsup-simcse-roberta-large
        3. princeton-nlp/sup-simcse-bert-large-uncased
        4. princeton-nlp/sup-simcse-roberta-large

        Permutation Case:
        1. default order ( d(x_i, x) < d(x_j, x) if i < j )
        2. reverse order ( d(x_i, x) > d(x_j, x) if i < j )
        3. Bootstrapping via Permutation
            3.1. Compute the majority voting of all (permutations) predictions.
            3.2. Compute the average of all permutations of sentence embeddings 
                and predict the sentence embedding that is closest to this average. 
                (Consider k!)

        Metric :
        1. euclidean
        2. cosine
    """
    def __init__(self, encoder_name, train_dataset, metric, logger, batch_size=128, device='cuda'):
        assert encoder_name in ["princeton-nlp/unsup-simcse-bert-large-uncased", "princeton-nlp/unsup-simcse-roberta-large",
                                "princeton-nlp/sup-simcse-bert-large-uncased", "princeton-nlp/sup-simcse-roberta-large"]
        assert metric in ["euclidean", "cosine"]

        if 'cuda' not in device:
            raise ValueError(f"Target device should be a gpu. Device {device} is not supported")
    
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.model = AutoModel.from_pretrained(encoder_name)
        self.encoder_name = encoder_name
        self.metric = metric
        self.logger = logger
        self.batch_size = batch_size

        self.model.to(device=device)# Use GPU Memory

        self.train_dataset = train_dataset
        train_dataloader = DataLoader([qa['question'] for qa in train_dataset], batch_size=batch_size)

        print ("Convert the questions in the train dataset to sentence embeddings. "); start = time.perf_counter()
        self.train_q_embeddings = []
        for train_i, batch in enumerate(tqdm(train_dataloader)):
            # Convert train (batch) questions to embeddings
            train_q_embeddings = self.convert_embedding(
                batch
            )
            self.train_q_embeddings.append(train_q_embeddings)
        self.train_q_embeddings = torch.cat(self.train_q_embeddings, dim=0)

        logger.info(f"Finish convert the train sentence embeddings : {time.perf_counter()-start} sec")
    
    def convert_embedding(self, inputs):
        # Tokenize input texts
        inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        input_data = {k: v.cuda() for k, v in inputs.items()}# "input_ids", "token_type_ids", "attention_mask"

        # Get the embeddings
        with torch.no_grad():
            # Get [CLS] token vector (classification token)
            input_embeddings = self.model(**input_data, output_hidden_states=True, return_dict=True).pooler_output

        return input_embeddings.detach().cpu()
    
    def upload_nearestneighbors(self, k, demo_seed, dataset_name):
        """
            encoder_name, demo_seed, n_retrieval_data, k

            ex) file name: {encoder_name} / {dataset_name} / seed({demo_seed})n({n_retrieval_data})k((k})
        """
        file_name = f"nbrs_{self.encoder_name}/{dataset_name}/seed({demo_seed})n({len(self.train_dataset)})k({k}).pkl"
        print (file_name)

        # Make Directory
        temp = file_name.split('/')
        for i in range(len(temp)):
            if temp[i].startswith('seed('): break
            dir = '/'.join(temp[:i+1])
            if not os.path.exists(dir):
                os.mkdir(dir)

        start = time.perf_counter()
        if os.path.exists(file_name):
            # Load the model from a file
            with open(file_name, 'rb') as f:
                nbrs = pickle.load(f)
            self.logger.info(f"Finish read Nearest Neighbors : {time.perf_counter()-start} sec")
        else:
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', n_jobs=-1).fit(self.train_q_embeddings)
            # Save the model to a file
            with open(file_name, 'wb') as f:
                pickle.dump(nbrs, f)
            self.logger.info(f"Finish write Nearest Neighbors : {time.perf_counter()-start} sec")

        return nbrs

    def select_k(self, k, test_dataset, demo_seed, dataset_name):
        assert self.metric in ["euclidean", "cosine"]
        if k==0: 
            return [[] for i in range(len(test_dataset))]

        if self.metric == "euclidean":
            nbrs = self.upload_nearestneighbors(k, demo_seed, dataset_name)
        else:
            raise NotImplementedError()

        test_dataloader = DataLoader([qa['question'] for qa in test_dataset], batch_size=self.batch_size)
        
        print ("Convert the questions in the test dataset to sentence embeddings. "); start = time.perf_counter()
        save_idx = []
        for test_i, test_batch in enumerate(tqdm(test_dataloader)):
            # Convert test questions to embeddings
            test_q_embeddings = self.convert_embedding(
                test_batch
            )
            
            if self.metric == "euclidean":
                distances, indices = nbrs.kneighbors(test_q_embeddings)
                for i in range(len(test_batch)):
                    save_idx.append({"index": torch.from_numpy(indices[i]), "value": distances[i]})

            elif self.metric == "cosine":
                # Calculate cosine similarities
                # Cosine similarities are in [-1, 1]. Higher means more similar (: Cosine Similarity, Not cosine similarity distance)
                cosine_similarities = pairwise.cosine_similarity(X=test_q_embeddings, Y=self.train_q_embeddings)
                for i in range(len(test_batch)):
                    # high_to_low
                    cosines, indices = torch.topk(torch.from_numpy(cosine_similarities[i]), k=k, dim=-1)
                    save_idx.append({"index": indices, "value": cosines})
                    """
                    Debugging
                        test_q_embeddings.shape, self.train_q_embeddings.shape, cosine_similarities.shape
                        cosines, indices
                        cosine_similarities[self.batch_size*test_i+i][indices]
                    """
            else:
                raise NotImplementedError()
        
        self.logger.info(f"Finish convert the test sentence embeddings : {time.perf_counter()-start} sec")
        return [dic["index"] for dic in save_idx]

        """
        Debugging
            a = [dic["index"] for dic in save_idx]
            for idx in a[0]: print(self.train_dataset[idx])
        """

    def majority_voting_prediction(self, all_permutation_outputs):
        """
            Bootstrapping via Permutation
            : Compute the majority voting of all (permutations) predictions.
        """
        if len(all_permutation_outputs) == 1: 
            return all_permutation_outputs[0]
        
        self.logger.info(f"All outputs : {all_permutation_outputs}")# [n_outputs], 4-shot: n_outputs==24

        output_counts = Counter(all_permutation_outputs)# {output1: n_output1, output2: n_output2, ...}
        sorted_output_counts = sorted(output_counts.items(), key=lambda x: x[1], reverse=True)
        
        self.logger.info(f"Count the Outputs : {sorted_output_counts}")
        self.logger.info(f"Prediction : {sorted_output_counts[0][0]}")

        return sorted_output_counts[0][0]

    def nearest_prediction(self, all_permutation_outputs):
        """
            Bootstrapping via Permutation
            : Compute the average of all permutations of sentence embeddings 
            and predict the sentence embedding that is closest to this average. 
            ( Consider k! )
        """# torch.Size([ batch_size, length ])
        if len(all_permutation_outputs) == 1: 
            return all_permutation_outputs[0]
        
        self.logger.info(f"All outputs : {all_permutation_outputs}")# [n_outputs], 4-shot: n_outputs==4!
        print ("Convert the predictions to sentence embeddings. "); start = time.perf_counter()

        print ("ZZZ : ", len(all_permutation_outputs))
        prediction_sentence_embeddings = []
        all_permutations_dataloader = DataLoader(all_permutation_outputs, batch_size=self.batch_size)

        for idx, batch in enumerate(tqdm(all_permutations_dataloader)):
            # Convert train (batch) questions to embeddings
            predict_embeddings = self.convert_embedding(
                batch
            )
            prediction_sentence_embeddings += predict_embeddings# 4-shot : torch.Size([24, 1024])
        mean_embedding = sum(prediction_sentence_embeddings)/len(prediction_sentence_embeddings)# torch.Size([1024])

        # Nearest Neighbors (n_neighbors=1)
        dist_list = []
        for embedding in prediction_sentence_embeddings:
            dist = distance.euclidean(mean_embedding, embedding)
            dist_list.append(dist)
        idx = dist_list.index(min(dist_list))

        print (f"mean_embedding : {mean_embedding} , prediction embedding : {prediction_sentence_embeddings[idx]}")
        self.logger.info(f"Prediction : {all_permutation_outputs[idx]} , distance : {dist_list[idx]}, index : {idx}")
        self.logger.info(f"Finish convert the prediction sentence embeddings : {time.perf_counter()-start} sec")

        return all_permutation_outputs[idx]

def ordering(train_dataset_indices, permu_seed=2023, ordering=None):
    """
    Ordering : Random Permutation (None), low_to_high, high_to_low, all_permutations_majority_voting, all_permutations_mean_embedding
    """
    assert ordering in [None, "low_to_high", "high_to_low", "all_permutations_majority_voting", "all_permutations_mean_embedding"]
    if len(train_dataset_indices) == 0: return []
    
    if ordering == None:
        np.random.seed(permu_seed)# Randomness about permu_seed
        indices = np.random.permutation(len(train_dataset_indices))
    elif ordering == "low_to_high":
        indices = range(len(train_dataset_indices)-1, -1, -1)
    elif ordering == "high_to_low":
        indices = range(len(train_dataset_indices))
    elif ordering.startswith("all_permutations"):
        indices = range(len(train_dataset_indices))
    else:
        raise NotImplementedError()
    
    return [ train_dataset_indices[idx] for idx in indices ]
