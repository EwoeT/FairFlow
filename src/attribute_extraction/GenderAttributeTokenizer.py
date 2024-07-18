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
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset


# random.seed(42)
torch.manual_seed(42) 
if torch.cuda.is_available():
  device = torch.device("cuda")
  print('There are %d GPU(s) available.' % torch.cuda.device_count())
  print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
  print('No GPU available, using the CPU instead.')
  device = torch.device("cpu")

# with open('../../DIIN/data/intrinsic_word_level_data/enwiki-20230320-pages-articles4.txt') as file:
#     lines = [line.rstrip() for line in file]
# attribute_df = pd.DataFrame(lines)[0]
# attribute_df.reset_index(drop=True)
# attribute_df


# attribute_1 = "catholic"
# female_attribute_list = [attribute_1]

# attribute_2 = "muslim"
# male_attribute_list = [attribute_2]

class tokenizer_config:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


class corpus_tokenizer(tokenizer_config):
    def __init__(self, model_name, attribute_list1, attribute_list2, seq_len):
        super().__init__(model_name)
        self.attribute_list1 = attribute_list1
        self.attribute_list2 = attribute_list2
        self.seq_len = seq_len
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # test_input_ids, test_target_labels, test_attention_masks = tokenize(X_test)
        
        
        # from transformers import BertTokenizer, AdamW, BertConfig, BertForPreTraining
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        vocabs = self.tokenizer.get_vocab()
        # seq_len = 150
    
    
    def tokenize_text(self, input_text):
        input_encoded_dict = self.tokenizer.batch_encode_plus(
                            list(input_text),                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = self.seq_len,           # Pad & truncate all sentences.
                            truncation=True,
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )
        return input_encoded_dict.input_ids, input_encoded_dict.attention_mask



    def get_corpus_token_ids(self, attribute_df):
        input_batch = list(attribute_df)[:int(0.9*len(attribute_df))]
        val_input_batch = list(attribute_df)[int(0.9*len(attribute_df)):]
        
        input_ids, input_attention_mask = self.tokenize_text(input_batch)
        val_input_ids, val_input_attention_mask = self.tokenize_text(val_input_batch)
        
        train_dataset_1 = TensorDataset(input_ids, input_attention_mask)
        val_dataset_1 = TensorDataset(val_input_ids, val_input_attention_mask)
        return train_dataset_1, val_dataset_1
    
    # torch.save(train_dataset_1, "1_train_dataset3.pt")
    # torch.save(val_dataset_1, "1_val_dataset3.pt")
    
    # train_dataset_1 = torch.load("1_train_dataset1.pt")
    # val_dataset_1 = torch.load("1_val_dataset1.pt")
    
    
    
class get_attribute_tokens(tokenizer_config):
    def __init__(self, model_name, attribute_list1, attribute_list2, train_dataset_1, val_dataset_1, seq_len):
        super().__init__(model_name)
        self.attribute_list1 = attribute_list1
        self.attribute_list2 = attribute_list2
        self.train_dataset_1 = train_dataset_1
        self.val_dataset_1 = val_dataset_1
        self.seq_len = seq_len

    
    def get_attributes_ids(self, gender_attributes):
        gender_attributes_ids = []
        for j in gender_attributes:
            wordpieces = self.tokenizer(j).input_ids
            if len(wordpieces)==3:
                gender_attributes_ids.append(wordpieces[1])
        return gender_attributes_ids

    
    def tokenize(self, tokenized_data, max_token_count):
        # Load the BERT tokenizer.
        print('Loading BERT tokenizer...')
    
        # Training Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []
        stereotype_token_id = []
        stereotype_token_index = []
        gender_label = []
        attention_masks = []
        count = {}
        

        train_female_attributes_ids = self.get_attributes_ids(self.attribute_list1)
        train_male_attributes_ids = self.get_attributes_ids(self.attribute_list2)
        print(train_female_attributes_ids, "\n")
        print(train_male_attributes_ids)
        
        for i in train_female_attributes_ids+train_male_attributes_ids:
            count[i] = 0
        
        # For every sentence...
        for k, sent in enumerate(tokenized_data):
            tokens_ids = sent[0].cpu().numpy().copy()
            att_masks = sent[1].cpu().numpy().copy()
            # print(tokens_ids)
            
    
            #         target_output = tokens.copy()
            if (count[train_female_attributes_ids[0]]>= max_token_count) and (count[train_male_attributes_ids[0]]>= max_token_count):
                print("iii")
                break
            else:
                for i, token in enumerate(tokens_ids):
                    if token in train_female_attributes_ids:
                        if count[token]<max_token_count: #prevent over imbalance of tokens
                            tokens_tensor = torch.tensor([tokens_ids])
                            input_ids.append(tokens_tensor)
                            attention_masks.append(torch.tensor([att_masks]))
                            stereotype_token_id.append(token)
                            stereotype_token_index.append(i)
                            gender_label.append(0)
                            count[token] += 1
                    elif token in train_male_attributes_ids:
                        if count[token]<max_token_count:
                            tokens_tensor = torch.tensor([tokens_ids])
                            input_ids.append(tokens_tensor)
                            attention_masks.append(torch.tensor([att_masks]))
                            stereotype_token_id.append(token)
                            stereotype_token_index.append(i)
                            gender_label.append(1)
                            count[token] += 1        
            
        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
    #     print(input_ids)
        attention_masks = torch.cat(attention_masks, dim=0)
    #     print(attention_masks)
        stereotype_token_id = torch.tensor(stereotype_token_id)
        
        stereotype_token_index = np.array(stereotype_token_index)
        b = np.zeros((stereotype_token_index.size, self.seq_len))
        b[np.arange(stereotype_token_index.size),stereotype_token_index] = 1
        stereotype_token_index = torch.tensor(b.astype(int))
    #     print(stereotype_token_index)
        
        gender_label = torch.tensor(gender_label)
    
        return input_ids, attention_masks,  stereotype_token_id, stereotype_token_index, gender_label, count

    
    
    def generate_tokens(self):    
        # max_token_count = 100000
        input_ids = torch.tensor(np.array([]).astype(int))
        attention_masks = torch.tensor(np.array([]).astype(int))
        stereotype_token_id = torch.tensor(np.array([]).astype(int))
        stereotype_token_index = torch.tensor(np.array([]).astype(int))
        gender_label = torch.tensor(np.array([]).astype(int))
        
        val_input_ids = torch.tensor(np.array([]).astype(int))
        val_attention_masks = torch.tensor(np.array([]).astype(int))
        val_stereotype_token_id = torch.tensor(np.array([]).astype(int))
        val_stereotype_token_index = torch.tensor(np.array([]).astype(int))
        val_gender_label = torch.tensor(np.array([]).astype(int))
        
        
        for ii in range(1,4):
            max_token_count =3000 #20000
            input_idsii, attention_masksii,  stereotype_token_idii, stereotype_token_indexii, gender_labelii, count = self.tokenize(self.train_dataset_1, max_token_count = max_token_count)
            val_input_idsii, val_attention_masksii,  val_stereotype_token_idii, val_stereotype_token_indexii, val_gender_labelii, val_count = self.tokenize(self.val_dataset_1, max_token_count = int(max_token_count/10))
        
            input_ids = torch.cat((input_ids, input_idsii), 0)
            attention_masks = torch.cat((attention_masks, attention_masksii), 0)
            stereotype_token_id = torch.cat((stereotype_token_id, stereotype_token_idii), 0)
            stereotype_token_index = torch.cat((stereotype_token_index, stereotype_token_indexii), 0)
            gender_label = torch.cat((gender_label, gender_labelii), 0)
        
        
            val_input_ids = torch.cat((val_input_ids, val_input_idsii), 0)
            val_attention_masks = torch.cat((val_attention_masks, val_attention_masksii), 0)
            val_stereotype_token_id = torch.cat((val_stereotype_token_id, val_stereotype_token_idii), 0)
            val_stereotype_token_index = torch.cat((val_stereotype_token_index, val_stereotype_token_indexii), 0)
            val_gender_label = torch.cat((val_gender_label, val_gender_labelii), 0)
            
            # print('Original: ', X_train[0])
            print('input_ids: ', input_ids[0], "\n")
            print('gender_label:', gender_label[0], "\n")
            print('stereotype_token_id: ', stereotype_token_id[0], "\n")
            print('stereotype_token_index:', stereotype_token_index[0], "\n")
            print('gender_label:', gender_label[0], "\n")
            
            
            
            train_dataset_2 = TensorDataset(input_ids, attention_masks,  stereotype_token_id, stereotype_token_index, gender_label)
            val_dataset_2 = TensorDataset(val_input_ids, val_attention_masks,  val_stereotype_token_id, val_stereotype_token_index, val_gender_label)
            # test_dataset = TensorDataset(test_input_ids, test_target_labels, test_attention_masks)
            
            print('{:>5,} training samples'.format(len(train_dataset_2)))
            print('{:>5,} validation samples'.format(len(val_dataset_2)))
            print('count', count)
            print('validation count', val_count)
            # print('{:>5,} test samples'.format(len(test_dataset)))
        
        
        # save tokenized dataset
        torch.save(train_dataset_2, f'outputs/2_train_{self.attribute_list1[0]}_{self.attribute_list2[0]}_tokens.pt')
        torch.save(val_dataset_2, f'outputs/2_val_{self.attribute_list1[0]}_{self.attribute_list2[0]}_tokens.pt')