# import tensorflow as tf
import numpy as np
import torch
import random
import pandas as pd
import os, sys
import time
import datetime
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from IPython.display import display, HTML
import os
import bert_model_2_extract_attributes_after_training
from transformers import AutoTokenizer
import time
import datetime
import pickle
from transformers.optimization import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import random
from tqdm.notebook import tqdm
from transformers.optimization import get_linear_schedule_with_warmup



# random.seed(42)
torch.manual_seed(42) 
if torch.cuda.is_available():
  device = torch.device("cuda")
  print('There are %d GPU(s) available.' % torch.cuda.device_count())
  print('We will use the GPU:', torch.cuda.get_device_name())

else:
  print('No GPU available, using the CPU instead.')
  device = torch.device("cpu")


class ExtractAttributeEmbeddings:
    def __init__(self, model_name, attribute_list1, attribute_list2, threshold, chunk_size, batch_size):
        self.model_name = model_name
        self.attribute_list1 = attribute_list1
        self.attribute_list2 = attribute_list2
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.attribute_pair = f"{self.attribute_list1[0]}_{self.attribute_list2[0]}"
        self.embeddings_path = f"outputs/attribute_embeddings_{self.attribute_pair}"

        if not os.path.exists(self.embeddings_path):
            os.makedirs(self.embeddings_path)
        
        self.BertForMaskedLM = bert_model_2_extract_attributes_after_training.BertForMaskedLM
        
        self.model = self.BertForMaskedLM.from_pretrained(self.model_name, self.attribute_pair, max_token_count=30, threshold=self.threshold)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.cuda()
           
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.vocabs = self.tokenizer.get_vocab()
        self.vv = dict((v,k) for k,v in self.vocabs.items())
        
    
    
    
    def format_time(self, elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))
    
    
    
    # Function to calculate the accuracy of our predictions vs labels
    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)
    
    
    
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
    def train(self, model, test_dataloader):
        seed_val = 42
    
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
    
        # We'll store a number of quantities such as training and validation loss, 
        # validation accuracy, and timings.
        training_stats = []
    
        # Measure the total training time for the whole run.
        total_t0 = time.time()
   
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
    
        print("")
        print("Running Extraction...")
    
        t0 = time.time()
    
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()
    
        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
    
        all_embeddings = []
        all_gen_labels = []
        all_attribute_token_id = []
        all_logits = []
        # Evaluate data for one epoch
        jj=0
    
        count_dict = {}
        for i in self.tokenizer.vocab.values():
            count_dict[i] = 0
    
        
        
        for batch in test_dataloader:
            jj=jj+1
            if jj%100==0:
                print(jj)
    
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
    
    

            with torch.no_grad():        
                # Perform a forward pass (evaluate the model on this training batch).
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # It returns different numbers of parameters depending on what arguments
                # arge given and what flags are set. For our useage here, it returns
                # the loss (because we provided labels) and the "logits"--the model
                # outputs prior to activation.
                embeddings, token_ids, gender, logits, count_dict = model(b_input_ids, 
                                     # token_type_ids=None,
                                     count_dict=count_dict,
                                     attention_mask=b_input_mask, 
                                     # labels=b_target_labels,
                                    )
    
                # print(len(res))
                # res = embeddings.cpu().detach().numpy()
                # print(len(res), len(b_target_labels))
                for key, iii in enumerate(embeddings):
                    all_embeddings.append(list(iii.cpu().detach().numpy()))
                    # print(len(all_embeddings))
                    all_attribute_token_id.append(token_ids[key].cpu().detach().numpy().item())
                    # print(all_attribute_token_id)
                    all_gen_labels.append(gender[key])
                    # print(all_gen_labels)
                    all_logits.append(logits[key].cpu().detach().numpy().item())
    
            # if jj<=20000:
                if jj%self.chunk_size==0:
                    # ccc = {vv[k]:v for k,v in count_dict.items()}
                    # print("ccc")
                    with open(self.embeddings_path+"/embeddings_"+str(jj)+".pkl", 'wb') as f:
                        pickle.dump(all_embeddings, f)
                    with open(self.embeddings_path+"/token_ids_"+str(jj)+".pkl", 'wb') as f:
                        pickle.dump(all_attribute_token_id, f)
                    with open(self.embeddings_path+"/gender_"+str(jj)+".pkl", 'wb') as f:
                        pickle.dump(all_gen_labels, f)
                    with open(self.embeddings_path+"/logits_"+str(jj)+".pkl", 'wb') as f:
                        pickle.dump(all_logits, f)
        
                    all_embeddings = []
                    all_gen_labels = []
                    all_attribute_token_id = []
                    all_logits = []


    

    def extract(self, train_dataset):
        train_dataloader = DataLoader(
                    train_dataset, # The validation samples.
                    sampler = SequentialSampler(train_dataset), # Pull out batches sequentially.
                    batch_size = self.batch_size # Evaluate with this batch size.
                    )
        self.train(self.model, train_dataloader)    
