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
import bert_model
import importlib
import time
import datetime
from transformers.optimization import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import random
from tqdm.notebook import tqdm
from transformers.optimization import get_linear_schedule_with_warmup




# random.seed(42)
torch.manual_seed(42) 
if torch.cuda.is_available():
  device = torch.device("cuda")

else:
  device = torch.device("cpu")



class AttributeClassifer:
    def __init__(self, model_name, attribute_list1, attribute_list2, batch_size, epochs):
        self.model_name = model_name
        self.attribute_1 = attribute_list1[0]
        self.attribute_2 = attribute_list2[0]
        self.train_dataset = torch.load(f'outputs/2_train_'+self.attribute_1+'_'+self.attribute_2+'_tokens.pt')
        self.val_dataset = torch.load(f'outputs/2_val_'+self.attribute_1+'_'+self.attribute_2+'_tokens.pt')
        self.BertForMaskedLM = bert_model.BertForMaskedLM
        self.model = self.BertForMaskedLM.from_pretrained(self.model_name)
        self.model.cuda()

        # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
        self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                          lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                        )
        
        
        
        
        # The DataLoader needs to know our batch size for training, so we specify it 
        # here. For fine-tuning BERT on a specific task, the authors recommend a batch 
        # size of 16 or 32.
        self.batch_size = batch_size
        
        # Create the DataLoaders for our training and validation sets.
        # We'll take training samples in random order. 
        self.train_dataloader = DataLoader(
                    self.train_dataset,  # The training samples.
                    sampler = RandomSampler(self.train_dataset), # Select batches randomly
                    batch_size = self.batch_size # Trains with this batch size.
                )
        
        # For validation the order doesn't matter, so we'll just read them sequentially.
        self.validation_dataloader = DataLoader(
                    self.val_dataset, # The validation samples.
                    sampler = SequentialSampler(self.val_dataset), # Pull out batches sequentially.
                    batch_size = self.batch_size # Evaluate with this batch size.
                )
        
        
        
        
        # Number of training epochs. The BERT authors recommend between 2 and 4. 
        # We chose to run for 4, but we'll see later that this may be over-fitting the
        # training data.
        self.epochs = epochs
        
        # Total number of training steps is [number of batches] x [number of epochs]. 
        # (Note that this is not the same as the number of training samples).
        self.total_steps = len(self.train_dataloader) * self.epochs
        
        # Create the learning rate scheduler.
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = self.total_steps)
    
    
    
    
    def format_time(self, elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))
        
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))
    
    # import numpy as np
    
    
    # Function to calculate the accuracy of our predictions vs labels
    def flat_accuracy(self, preds, labels):
        # print(preds)
        # pred_flat = np.argmax(preds, axis=1).flatten()
        pred_flat = torch.round((torch.sign(preds) + 1) / 2)
        pred_flat = np.array(pred_flat.squeeze().long().tolist())
        labels_flat = np.array(labels.squeeze().tolist())
        # print("pred", pred_flat,"\n")
        # print("labl", labels_flat,"\n")
        return np.sum(pred_flat == labels_flat) / len(labels_flat)
    
    
    
    
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
    def train(self, model, losslogger, start_epoch, epochs, run_id, train_dataloader, validation_dataloader, checkpoint_name):
        # Set the seed value all over the place to make this reproducible.
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
    
        # For each epoch...
        t = tqdm(range(start_epoch,epochs))
    
        for epoch_i in t:
    
            # ========================================
            #               Training
            # ========================================
    
            # Perform one full pass over the training set.
    
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')
    
            # Measure how long the training epoch takes.
            t0 = time.time()
    
            # Reset the total loss for this epoch.
            total_train_loss = 0
    
            # Put the model into training mode. Don't be mislead--the call to 
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            model.train()
    
            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):
    
                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = self.format_time(time.time() - t0)
    
                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
    
    
                # Unpack this training batch from our dataloader. 
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the 
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                #   [1]: attention masks
                #   [2]: labels 
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                stereotype_token_index = batch[3].to(device)
                b_target_labels = batch[4].to(device)
                
    #             #     convert input ids to one hot encoding
    #             b_input_ids = b_input_ids.reshape(len(b_input_ids), seq_len, 1)
    #             y_onehot = torch.FloatTensor(len(b_input_ids), seq_len, vocab_length)
    #             y_onehot.zero_()
    #             y_onehot.scatter_(2, b_input_ids, 1)
    #             y_onehot = y_onehot.float().to(device)
    
                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because 
                # accumulating the gradients is "convenient while training RNNs". 
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                model.zero_grad()        
    
                # Perform a forward pass (evaluate the model on this training batch).
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # It returns different numbers of parameters depending on what arguments
                # arge given and what flags are set. For our useage here, it returns
                # the loss (because we provided labels) and the "logits"--the model
                # outputs prior to activation.
                outputs = model(b_input_ids, 
                                     token_type_ids=None,
                                     stereotype_token_index=stereotype_token_index,
                                     attention_mask=b_input_mask, 
                                     labels=b_target_labels,
                                    )
    
                loss, logits = outputs.loss, outputs.logits
                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value 
                # from the tensor.
    #             print(outputs)
                total_train_loss += loss.item()
    
                # Perform a backward pass to calculate the gradients.
                loss.backward()
    
                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                self.optimizer.step()
    
                # Update the learning rate.
                self.scheduler.step()
    
            if epoch_i%1==0:
                torch.save(model, f"outputs/classification_model_{self.attribute_1}_{self.attribute_2}_{str(epoch_i)}.pth")
            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)            
    
            # Measure how long this epoch took.
            training_time = self.format_time(time.time() - t0)
            
            # set in logs
            df = pd.DataFrame()
            df['chackpoint_name'] = pd.Series(checkpoint_name)
            df['epoch'] = pd.Series(epoch_i)
            df['Loss'] = pd.Series(loss.data.item())
            df['run'] = run_id
            # losslogger = losslogger.append(df)
            
            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))
            state = {'epoch': epoch_i + 1, 'state_dict': model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'losslogger': losslogger, }
    #         torch.save(state, f'{checkpoint_name}')
    
            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.
    
            print("")
            print("Running Validation...")
    
            t0 = time.time()
    
            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()
    
            # Tracking variables 
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0
    
            # Evaluate data for one epoch
            for batch in validation_dataloader:
    
                # Unpack this training batch from our dataloader. 
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using 
                # the `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                #   [1]: attention masks
                #   [2]: labels 
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                stereotype_token_index = batch[3].to(device)
                b_target_labels = batch[4].to(device)
                
    
                
                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():        
    
                    # Forward pass, calculate logit predictions.
                    # token_type_ids is the same as the "segment ids", which 
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here: 
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    # Get the "logits" output by the model. The "logits" are the output
                    # values prior to applying an activation function like the softmax.
                    outputs = model(b_input_ids, 
                                         token_type_ids=None,
                                         stereotype_token_index=stereotype_token_index,
                                         attention_mask=b_input_mask, 
                                         labels=b_target_labels,
                                        )
                
                loss, logits = outputs.loss, outputs.logits
                # Accumulate the validation loss.
                total_eval_loss += loss.item()
    
                # Move logits and labels to CPU
                logits = logits.reshape(len(b_target_labels)).detach().cpu()
    #             print(logits.shape)
                label_ids = b_target_labels.to('cpu')
    #             print(label_ids.shape)
    
                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_eval_accuracy += self.flat_accuracy(logits, label_ids)
    #             print(total_eval_accuracy)
    
    
            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    #         print(total_eval_accuracy)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    
            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)
    
            # Measure how long the validation run took.
            validation_time = self.format_time(time.time() - t0)
    
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))
    
            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )
    
    #     print("")
    #     print("Training complete!")
    
              
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
    
    
    def train_model(self):         
        # first training epoch
        # Start
        start_epoch = 0
        
        # Logger
        losslogger = pd.DataFrame()
        
        # Checkpoint name
        checkpoint_name = 'checkpoint.pth.tar'
        
        self.train(self.model, losslogger, start_epoch, self.epochs, 0, self.train_dataloader, self.validation_dataloader, checkpoint_name)
        time.sleep(8)          
              
