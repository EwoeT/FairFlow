import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, BartConfig, BertTokenizer, BertForSequenceClassification, BartTokenizer
# from BART_dual_obj import BartForConditionalGeneration
from transformers import BartForConditionalGeneration
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset
import torch.optim as optim
import random
import torch.nn as nn
from tqdm.notebook import tqdm
import time
import datetime
from transformers.optimization import AdamW    
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

if torch.cuda.is_available():
  device = torch.device("cuda")
  print('There are %d GPU(s) available.' % torch.cuda.device_count())
  print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
  print('No GPU available, using the CPU instead.')
  device = torch.device("cpu")
    


class adv_model_train():
    def __init__(self, bart_pretrained_version):
        super().__init__()
        self.bart_pretrained_version = bart_pretrained_version
        self.config = BartConfig()
        self.model_F = BartForConditionalGeneration.from_pretrained(self.bart_pretrained_version)
        self.model_F.cuda()
        self.num_labels = 2
        self.criterion = nn.BCELoss()
        self.seq_len = 100
        self.tokenizer = BartTokenizer.from_pretrained(self.bart_pretrained_version)
    
    def data_prep(self, real_input_batch, input_batch, labels_batch, val_real_input_batch, val_input_batch, val_labels_batch):
        # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        tokenizer = AutoTokenizer.from_pretrained(self.bart_pretrained_version)
        seq_len = 150
        real_input_ids = tokenizer.batch_encode_plus(real_input_batch, max_length = seq_len, truncation=True, return_tensors="pt", padding=True).input_ids
        input_ids = tokenizer.batch_encode_plus(input_batch, max_length = seq_len, truncation=True, return_tensors="pt", padding=True).input_ids
        labels_ids = tokenizer.batch_encode_plus(labels_batch, max_length = seq_len, truncation=True, return_tensors="pt", padding=True).input_ids
        val_real_input_ids = tokenizer.batch_encode_plus(val_real_input_batch, max_length = seq_len, truncation=True, return_tensors="pt", padding=True).input_ids
        val_input_ids = tokenizer.batch_encode_plus(val_input_batch, max_length = seq_len, truncation=True, return_tensors="pt", padding=True).input_ids
        val_labels_ids = tokenizer.batch_encode_plus(val_labels_batch, max_length = seq_len, truncation=True, return_tensors="pt", padding=True).input_ids

        train_dataset = TensorDataset(real_input_ids, input_ids, labels_ids)
        val_dataset = TensorDataset(val_real_input_ids, val_input_ids, val_labels_ids)
        print('{:>5,} training samples'.format(len(train_dataset)))
        print('{:>5,} validation samples'.format(len(val_dataset)))
        return train_dataset, val_dataset
    


    def data_prep_model_F_training_data(self, real_input_batch, labels_batch, gen_labels_batch):
        # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        tokenizer = AutoTokenizer.from_pretrained(self.bart_pretrained_version)
        seq_len = 200
        labels_ids = tokenizer.batch_encode_plus(labels_batch, max_length = seq_len, truncation=True, padding=True, return_tensors="pt").input_ids
        real_input_ids = tokenizer.batch_encode_plus(real_input_batch, max_length = seq_len, truncation=True, return_tensors="pt", padding=True).input_ids
        gen_labels = torch.tensor(gen_labels_batch)

        train_dataset = TensorDataset(real_input_ids, labels_ids, gen_labels)
        # val_dataset = TensorDataset(val_real_input_ids, val_input_ids, val_labels_ids)
        print('{:>5,} training samples'.format(len(train_dataset)))
        # print('{:>5,} validation samples'.format(len(val_dataset)))
        return train_dataset


    ############################ train model ##################################


    vocab_length = 30522



    def all_parameters(self, train_dataset, val_dataset):
        start_epoch = 0
        batch_size = 24
        train_dataloader = DataLoader(
                    train_dataset,  # The training samples.
                    sampler = RandomSampler(train_dataset), # Select batches randomly
                    batch_size = batch_size # Trains with this batch size.
                )
        validation_dataloader = DataLoader(
                    val_dataset, # The validation samples.
                    sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                    batch_size = batch_size # Evaluate with this batch size.
                )
        epochs = 10
        return start_epoch, epochs, train_dataloader, validation_dataloader




    def format_time(self,elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    # import numpy as np


    # Function to calculate the accuracy of our predictions vs labels
    def flat_accuracy(self,preds, labels):
    #     pred_score = preds
        logits_argmax = np.array([np.argmax(l, axis=1) for l in preds])
        pred_flat = logits_argmax.flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)


    # Function to calculate the accuracy of our predictions vs labels
    def flat_cls_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)


    def train_modelF(self, train_dataset, val_dataset, batch_size):
        start_epoch = 0
        epochs = 10
        # batch_size = batch_size
        train_dataloader = DataLoader(
                    train_dataset,  # The training samples.
                    # sampler = RandomSampler(train_dataset), # Select batches randomly
                    batch_size = batch_size # Trains with this batch size.
                )
        validation_dataloader = DataLoader(
                    val_dataset, # The validation samples.
                    # sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                    batch_size = batch_size # Evaluate with this batch size.
                )
        total_steps = len(train_dataloader) * epochs
        optimizerF = AdamW(filter(lambda p: p.requires_grad, self.model_F.parameters()),
                  lr = 5e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                ) 
        schedulerF = get_linear_schedule_with_warmup(optimizerF, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
        # For each epoch...
        t = tqdm(range(start_epoch,epochs))
        for epoch_i in t:
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            total_train_lossF = 0

            self.model_F.train()
            for step, batch in enumerate(train_dataloader):
                if step % 40 == 0 and not step == 0:
                    elapsed = self.format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                    if step % 1200 == 0 and not step == 0:
                        torch.save(self.model_F, "trained_model_try_"+str(epoch_i)+".pth")
                b_real_input_ids = batch[0].to(device)
                b_labels = batch[1].to(device)
                b_labels_2 = batch[2].to(device)
                self.model_F.zero_grad()
                outputsF = self.model_F(input_ids=b_real_input_ids, labels=b_labels, output_hidden_states=True)
                errF = outputsF.loss
                errF.backward()
                torch.nn.utils.clip_grad_norm_(self.model_F.parameters(), 1.0)
                optimizerF.step()
                schedulerF.step()
                total_train_lossF += errF.item()
            torch.save(self.model_F, "trained_model_"+str(epoch_i)+".pth")
            avg_train_lossF = total_train_lossF / len(train_dataloader)
            # Measure how long this epoch took.
            training_time = self.format_time(time.time() - t0)

            
            print("")
            print("  Average training loss avg_train_lossF {0:.2f}".format(avg_train_lossF))
            print("  Training epcoh took: {:}".format(training_time))
        torch.save(self.model_F, "trained_model.pth")

                
        return self.model_F

        
    def load_checkpoint(model, optimizer, losslogger, filename):
        # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
        start_epoch = 0
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            losslogger = checkpoint['losslogger']
            print("=> loaded checkpoint '{}' (epoch {})"
                      .format(filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(filename))
    
        return model, optimizer, start_epoch, losslogger  
    
    
    
  



