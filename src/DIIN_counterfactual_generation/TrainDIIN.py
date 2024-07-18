from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from TransformerGlow import AdamWeightDecayOptimizer, FactorTrainer
import numpy as np
import torch
import random
import pandas as pd
import pickle
import os


# random.seed(42)
torch.manual_seed(42) 
if torch.cuda.is_available():
  device = torch.device("cuda")
  print('There are %d GPU(s) available.' % torch.cuda.device_count())
  print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
  print('No GPU available, using the CPU instead.')
  device = torch.device("cpu")



class TrainDIIN:
    # def __init__(self):
    #     self.n_factors = n_factors
    #     self.in_channel = in_channel
    #     self.n_flow = n_flow
    #     self.hidden_depth = hidden_depth
    #     self.hidden_dim = hidden_dim
    #     self.rho = rho
        

    def get_examples(self, embeddings, gen_labels, attribute_ids, logits, rand_seed=42 , train=True):
        random.seed(rand_seed)
        female_embeddings = []
        male_embeddings = []
        female_gen_labels = []
        male_gen_labels = []
        female_attribute_ids = []
        male_attribute_ids = []
        
        for key,i in enumerate(gen_labels):
            if i==0:
                # print(logits[key])
                if logits[key]<-0:
                    female_embeddings.append(embeddings[key])
                    female_gen_labels.append(i)
                    female_attribute_ids.append(attribute_ids[key])
                    # print(key)
            else:
                if logits[key]>0:
                    male_embeddings.append(embeddings[key])
                    male_gen_labels.append(i)
                    male_attribute_ids.append(attribute_ids[key])
    
        random.shuffle(female_embeddings)
        random.shuffle(male_embeddings)
        female_embeddings_1 = female_embeddings[:int(len(female_embeddings)*0.5)].copy()
        male_embeddings_1 = male_embeddings[:int(len(male_embeddings)*0.5)].copy()
        female_embeddings_2 = female_embeddings[int(len(female_embeddings)*0.5):].copy()
        male_embeddings_2 = male_embeddings[int(len(male_embeddings)*0.5):].copy()
        # random.Random(rand_seed).shuffle(female_embeddings_2)
        # random.Random(rand_seed).shuffle(male_embeddings_2)
    
        female_data_pairs = [[x, y] for x, y in zip(female_embeddings_1, female_embeddings_2)]
        male_data_pairs = [[x, y] for x, y in zip(male_embeddings_1, male_embeddings_2)]
    
        gender_data_pairs = female_data_pairs + male_data_pairs
        random.shuffle(gender_data_pairs)
    
        
        
    
        if train==True:
            gender_data_pairs = torch.tensor(np.array(gender_data_pairs))
            return gender_data_pairs
    
        else:
            female_data_pairs_test = torch.tensor(np.array(female_data_pairs))
            male_data_pairs_test = torch.tensor(np.array(male_data_pairs))
            return female_data_pairs_test, male_data_pairs_test


    
    def train_model(self, attribute1_list, attribute2_list, n_factors, in_channel, n_flow, hidden_depth, hidden_dim, rho, batch_size, chunk_size):
        # self.attribute1_list = attribute1_list
        # self.attribute2_list = attribute2_list
        # self.n_factors = n_factors
        # self.in_channel = in_channel
        # self.n_flow = n_flow
        # self.hidden_depth = hidden_depth
        # self.hidden_dim = hidden_dim
        # self.rho = rho
        # self.batch_size = batch_size
        # print(attribute1_list, attribute2_list, n_factors, in_channel, n_flow, hidden_depth, hidden_dim, rho, batch_size)
        FactorTrainer_config = {
          "n_factors": n_factors,
          # "factor_dim":211,
          "in_channel": in_channel,
          "n_flow": n_flow,
          "hidden_depth": hidden_depth,
          "hidden_dim": hidden_dim,
          "rho": rho
        }
        
        bertflow = FactorTrainer(FactorTrainer_config).cuda()
        
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters= [
                {
                    "params": [p for n, p in bertflow.glow.named_parameters()  \
                                    if not any(nd in n for nd in no_decay)],  # Note only the parameters within bertflow.glow will be updated and the Transformer will be freezed during training.
                    "weight_decay": 0.01,
                },
                {
                    "params": [p for n, p in bertflow.glow.named_parameters()  \
                                    if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
        
        
        
        optimizer = AdamWeightDecayOptimizer(
                params=optimizer_grouped_parameters, 
                lr=1e-3, 
                eps=1e-6,
            )
        
        
        attribute_pair = f"{attribute1_list[0]}_{attribute2_list[0]}"
        model_path = "outputs/bertflow_model_"+attribute_pair
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        
        bertflow.train()
        
        # for jj in range(1,2):
        for iteration in range(chunk_size, chunk_size+1, chunk_size+1):
            print(iteration)
        
            with open('outputs/attribute_embeddings_'+attribute_pair+'/embeddings_'+str(iteration)+'.pkl', 'rb') as f:
                embeddings = pickle.load(f)
            with open('outputs/attribute_embeddings_'+attribute_pair+'/gender_'+str(iteration)+'.pkl', 'rb') as f:
                    gen_labels = pickle.load(f)
            with open('outputs/attribute_embeddings_'+attribute_pair+'/token_ids_'+str(iteration)+'.pkl', 'rb') as f:
                    attribute_ids = pickle.load(f)
            with open('outputs/attribute_embeddings_'+attribute_pair+'/logits_'+str(iteration)+'.pkl', 'rb') as f:
                    logits = pickle.load(f)

        
            for epoch in range(5):
                # reload data shuffled
                train_gender_data_pairs = self.get_examples(embeddings, gen_labels, attribute_ids, logits, rand_seed=epoch)
                
                
                batch_size = batch_size
                
                
                
                # Create the DataLoaders for our training and validation sets.
                # We'll take training samples in random order. 
                train_dataloader = DataLoader(
                            train_gender_data_pairs,  # The training samples.
                            sampler = RandomSampler(train_gender_data_pairs), # Select batches randomly
                            batch_size = batch_size # Trains with this batch size.
                        )
            
               
                print("epoch:", epoch)
                for step, batch in enumerate(train_dataloader):
                    # print(step)
                    # print(batch.shape)
                    # bertflow.train()
                    z, loss = bertflow(batch.to(device))  # Here z is the sentence embedding
                    if step%100==0:
                        print(loss)
                        print(loss, file=open('output.txt', 'a'))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
                # if (iteration)%2000==0:
                # # if (epoch+1)%4==0:
                #     torch.save(bertflow, model_path+'/bertflow_rho_0999_factors_6_iterations_'+str(iteration)+'.pth')
            torch.save(bertflow, model_path+'/bertflow.pth')