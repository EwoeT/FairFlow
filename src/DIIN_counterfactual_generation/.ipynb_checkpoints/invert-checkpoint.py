from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from TransformerGlow import AdamWeightDecayOptimizer, FactorTrainer
import numpy as np
import torch
import random
import pandas as pd
import pickle
from transformers import BertTokenizer, BertForMaskedLM
from torch.utils.data import TensorDataset
import copy



torch.manual_seed(42) 
if torch.cuda.is_available():
  device = torch.device("cuda")
  print('There are %d GPU(s) available.' % torch.cuda.device_count())
  print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
  print('No GPU available, using the CPU instead.')
  device = torch.device("cpu")



class invert:
    def __init__(self, model_name):
        # self.model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.vocabs = self.tokenizer.get_vocab()
        self.vv = dict((v,k) for k,v in self.vocabs.items())
        self.embedding_decoder = BertForMaskedLM.from_pretrained(model_name).cls.cuda()
    
    # def get_examples1(self, embeddings, gen_labels, attribute_ids, rand_seed=2, train=True):
    #     random.seed(rand_seed)
    #     female_embeddings = []
    #     male_embeddings = []
    #     female_gen_labels = []
    #     male_gen_labels = []
    #     female_attribute_ids = []
    #     male_attribute_ids = []
    #     female_logits = []
    #     male_logits = []
    
    #     # print(len(gen_labels))
    #     for key,i in enumerate(gen_labels):
    #         # print(i)
    #         if i==0:
    #             female_embeddings.append(embeddings[key])
    #             female_gen_labels.append(i)
    #             female_attribute_ids.append(attribute_ids[key])
    #             # female_logits.append(logits[key])
    #             # print(key)
    #         else:
    #             male_embeddings.append(embeddings[key])
    #             # print(embeddings[key])
    #             male_gen_labels.append(i)
    #             male_attribute_ids.append(attribute_ids[key])
    #             # male_logits.append(logits[key])
    
    #     female_embeddings = torch.tensor(np.array(female_embeddings))
    #     female_attribute_ids = torch.tensor(np.array(female_attribute_ids))
    #     female_gen_labels = torch.tensor(np.array(female_gen_labels))
    #     # female_logits = torch.tensor(np.array(female_logits))
        
    #     male_embeddings = torch.tensor(np.array(male_embeddings))
    #     male_attribute_ids = torch.tensor(np.array(male_attribute_ids))
    #     male_gen_labels = torch.tensor(np.array(male_gen_labels))
    #     # male_logits = torch.tensor(np.array(male_logits))
        
    #     return female_embeddings, female_attribute_ids, female_gen_labels, male_embeddings, male_attribute_ids, male_gen_labels
    
    
    
    def get_examples(self, embeddings, gen_labels, attribute_ids, logits, rand_seed, train=True):
        random.seed(rand_seed)
        female_embeddings = []
        male_embeddings = []
        female_gen_labels = []
        male_gen_labels = []
        female_attribute_ids = []
        male_attribute_ids = []
        female_logits = []
        male_logits = []
    
        # print(len(gen_labels))
        for key,i in enumerate(gen_labels):
            # print(i)
            if i==0:
                if logits[key]<-0:
                    female_embeddings.append(embeddings[key])
                    female_gen_labels.append(i)
                    female_attribute_ids.append(attribute_ids[key])
                    female_logits.append(logits[key])
                    # print(key)
            else:
                if logits[key]>0:
                    male_embeddings.append(embeddings[key])
                    # print(embeddings[key])
                    male_gen_labels.append(i)
                    male_attribute_ids.append(attribute_ids[key])
                    male_logits.append(logits[key])
    
        female_embeddings = torch.tensor(np.array(female_embeddings))
        female_attribute_ids = torch.tensor(np.array(female_attribute_ids))
        female_gen_labels = torch.tensor(np.array(female_gen_labels))
        female_logits = torch.tensor(np.array(female_logits))
        
        male_embeddings = torch.tensor(np.array(male_embeddings))
        male_attribute_ids = torch.tensor(np.array(male_attribute_ids))
        male_gen_labels = torch.tensor(np.array(male_gen_labels))
        male_logits = torch.tensor(np.array(male_logits))
        
        return female_embeddings, female_attribute_ids, female_gen_labels, female_logits, male_embeddings, male_attribute_ids, male_gen_labels, male_logits
    
    
    

    def decode_emb(self, emb):
        self.embedding_decoder.eval()
        with torch.no_grad():
            logits_list = self.embedding_decoder(torch.tensor(emb))
            id_list = torch.argmax(logits_list, dim=1)
            return id_list
    
    


    def gen_flow_embs1_a(self, male_dataloader,bertflow):
        bertflow.eval()
        
        female_z_list = []
        male_z_list = []
        f_embs = []
        m_embs = []
        f_ids = []
        m_ids = []
        f_logits = []
        m_logits = []
        
        # For each batch of training data...
        bertflow.eval()
        with torch.no_grad():
            # for step, f_batch in enumerate(female_dataloader):
            #     # print(f_batch.shape)
            #     embs = f_batch[0]
            #     ids = f_batch[1]
            #     logit_v = f_batch[3]
            #     # print(embs.shape)
            #     female_z = bertflow(embs.to(device), return_loss=False)  # Here z is the sentence embedding
            #     female_z_list.append(female_z)
            #     f_embs.append(embs)
            #     f_ids.append(ids)
            #     f_logits.append(logit_v)
            for step, m_batch in enumerate(male_dataloader):
                embs = m_batch[0]
                ids = m_batch[1]
                logit_v = m_batch[3]
                # gen = m_batch[2]
                male_z = bertflow(embs.to(device), return_loss=False)  # Here z is the sentence embedding
                male_z_list.append(male_z)
                m_embs.append(embs)
                m_ids.append(ids)
                m_logits.append(logit_v)
    
        new_female_z_list = copy.deepcopy(female_z_list)
        # new_male_z_list = copy.deepcopy(male_z_list)
    
        # gender_transfer_count = 0
        # semantic_transfer_count = 0
        # non_conversion_count = 0
        # original_old_word = []
        # old_word = []
        # old_id = []
        # new_word = []
        # new_id = []
        # gender = []
        # logit_vals = []
    
        sum_fem = 0
        count = 0
        for j in range(len(male_z_list)):
            # print("############", j)
            for k in range(len(male_z_list[j][0])):
                sum_fem = sum_fem + male_z_list[j][0][k]
                count+=1
        avg_sum_fem = sum_fem/count
        return avg_sum_fem


    
    
    #########generate embeddings in gaussian space#################
    def gen_flow_embs_a(self, female_dataloader, male_dataloader, avg_sum_fem, bertflow):
        bertflow.eval()
        
        female_z_list = []
        male_z_list = []
        f_embs = []
        m_embs = []
        f_ids = []
        m_ids = []
        f_logits = []
        m_logits = []
        
        # For each batch of training data...
        bertflow.eval()
        with torch.no_grad():
            for step, f_batch in enumerate(female_dataloader):
                # print(f_batch.shape)
                embs = f_batch[0]
                ids = f_batch[1]
                logit_v = f_batch[3]
                # print(embs.shape)
                female_z = bertflow(embs.to(device), return_loss=False)  # Here z is the sentence embedding
                female_z_list.append(female_z)
                f_embs.append(embs)
                f_ids.append(ids)
                f_logits.append(logit_v)
            # for step, m_batch in enumerate(male_dataloader):
            #     embs = m_batch[0]
            #     ids = m_batch[1]
            #     logit_v = m_batch[3]
            #     # gen = m_batch[2]
            #     male_z = bertflow(embs.to(device), return_loss=False)  # Here z is the sentence embedding
            #     male_z_list.append(male_z)
            #     m_embs.append(embs)
            #     m_ids.append(ids)
            #     m_logits.append(logit_v)
    
        new_female_z_list = copy.deepcopy(female_z_list)
        # new_male_z_list = copy.deepcopy(male_z_list)
    
        gender_transfer_count = 0
        semantic_transfer_count = 0
        non_conversion_count = 0
        original_old_word = []
        old_word = []
        old_id = []
        new_word = []
        new_id = []
        gender = []
        logit_vals = []
    
        # sum_fem = 0
        # count = 0
        # for j in range(len(male_z_list)):
        #     # print("############", j)
        #     for k in range(len(male_z_list[j][0])):
        #         sum_fem = sum_fem + male_z_list[j][0][k]
        #         count+=1
        # avg_sum_fem = sum_fem/count
    
        for j in range(len(female_z_list)):
            for k in range(len(female_z_list[j][0])):
            # part1 = torch.tensor([female_z[0][0][15].cpu().detach().numpy() for i in female_z[0][0]])
                # new_male_z_list[j][0][0][k] = female_z_list[j][0][0][k]
                new_female_z_list[j][0][k] = avg_sum_fem
            
                
            eee = bertflow(new_female_z_list[j], reverse=True)
            eee = torch.squeeze(eee)
            ffff = self.decode_emb(eee)
            ffff
            for key,i in enumerate(self.decode_emb(f_embs[j][:,:].to(device))):
                # if f_logits[j][key].item() < -10:
                original_old_word.append(self.vv[f_ids[j][key].item()])
                old_word.append(self.vv[i.item()])
                old_id.append(i.item())
                new_word.append(self.vv[ffff[key].item()])
                new_id.append(ffff[key].item())
                gender.append(1)
                logit_vals.append(f_logits[j][key].item())
        
        return original_old_word, old_word, old_id, new_word, new_id, gender, logit_vals
    
    
    
    
    
    
    
    
    
    def get_instances_a(self, attribute1_list, attribute2_list, chunk_size):
        attribute_pair = f"{attribute1_list[0]}_{attribute2_list[0]}"
        model_path = "outputs/bertflow_model_"+attribute_pair
        bertflow = torch.load(f"{model_path}/bertflow.pth")
        original_old_word = []
        old_word = []
        old_id = []
        new_word = []
        new_id = []
        gender = []
        logit_vals = []
        
        
        for jj in range(1,2):
            for iteration in range(chunk_size,chunk_size+1,chunk_size+1):
                print(iteration)
                with open('outputs/attribute_embeddings_'+attribute_pair+'/embeddings_'+str(iteration)+'.pkl', 'rb') as f:
                    embeddings = pickle.load(f)
                with open('outputs/attribute_embeddings_'+attribute_pair+'/gender_'+str(iteration)+'.pkl', 'rb') as f:
                        gen_labels = pickle.load(f)
                with open('outputs/attribute_embeddings_'+attribute_pair+'/token_ids_'+str(iteration)+'.pkl', 'rb') as f:
                        attribute_ids = pickle.load(f)
                with open('outputs/attribute_embeddings_'+attribute_pair+'/logits_'+str(iteration)+'.pkl', 'rb') as f:
                        logits = pickle.load(f)
        
            
                start_ind = 0
                end_ind = 0
                # ratio = 10000
                ratio = len(attribute_ids)
                
                for i in range(int(len(attribute_ids)/ratio)):
                    print(i)
                    end_ind+=ratio
                    female_embeddings, female_attribute_ids, female_gen_labels, female_logits, male_embeddings, male_attribute_ids, male_gen_labels, male_logits = self.get_examples(embeddings[start_ind:end_ind], gen_labels[start_ind:end_ind], attribute_ids[start_ind:end_ind], logits[start_ind:end_ind], 1, train=False)
                    female_data_pairs_test = TensorDataset(female_embeddings, female_attribute_ids, female_gen_labels, female_logits)
                    male_data_pairs_test = TensorDataset(male_embeddings, male_attribute_ids, male_gen_labels, male_logits)
                    
                    female_dataloader = DataLoader(
                                female_data_pairs_test,  # The training samples.
                                sampler = None, #RandomSampler(female_data_pairs_test), # Select batches randomly
                                batch_size = 20 # Trains with this batch size.
                            )
                    
                    male_dataloader = DataLoader(
                                male_data_pairs_test,  # The training samples.
                                sampler = None, #RandomSampler(male_data_pairs_test), # Select batches randomly
                                batch_size = 20 # Trains with this batch size.
                            )
                    start_ind += ratio
                    # try:
                    # print((female_embeddings.shape), (male_embeddings.shape))
                    avg_sum_fem = self.gen_flow_embs1_a(male_dataloader, bertflow)
                    original_old_word_temp, old_word_temp, old_id_temp, new_word_temp, new_id_temp, gender_temp, logit_vals_temp = self.gen_flow_embs_a(female_dataloader, male_dataloader, avg_sum_fem, bertflow)
                    original_old_word+= original_old_word_temp
                    old_word+= old_word_temp
                    old_id+= old_id_temp
                    new_word+= new_word_temp
                    new_id+= new_id_temp
                    gender+= gender_temp
                    logit_vals+= logit_vals_temp
                    # except:
                    #     original_old_word_temp, old_word_temp, old_id_temp, new_word_temp, new_id_temp, gender_temp, logit_vals_temp
                    #     print(len(male_dataloader), len(female_dataloader))
            
        
        
        dict = {'original_old_word': original_old_word, 'old_word': old_word, 'old_id': old_id, 'new_word': new_word, 'new_id': new_id, 'gender': gender, 'logits': logit_vals}
        df = pd.DataFrame(dict)
        df["count"] = 1 
        df
        
        
        filter = df['original_old_word'].str.contains("##")
        df = df[~filter]
        filter = df['original_old_word'].str.contains("[CLS]")
        df = df[~filter]
        filter = df['original_old_word'].str.contains("[SEP]")
        df = df[~filter]
        filter = df['original_old_word'].str.contains("[PAD]")
        df = df[~filter]
        
        
        df.to_csv('outputs/female_male_full_pairs_'+attribute_pair+'.csv', index=False)






##################################b################
    def gen_flow_embs1_b(self, female_dataloader, bertflow):
        bertflow.eval()
        
        female_z_list = []
        male_z_list = []
        f_embs = []
        m_embs = []
        f_ids = []
        m_ids = []
        f_logits = []
        m_logits = []
        
        # For each batch of training data...
        bertflow.eval()
        with torch.no_grad():
            for step, f_batch in enumerate(female_dataloader):
                # print(f_batch.shape)
                embs = f_batch[0]
                ids = f_batch[1]
                logit_v = f_batch[3]
                # print(embs.shape)
                female_z = bertflow(embs.to(device), return_loss=False)  # Here z is the sentence embedding
                female_z_list.append(female_z)
                f_embs.append(embs)
                f_ids.append(ids)
                f_logits.append(logit_v)
            # for step, f_batch in enumerate(female_dataloader):
            #     embs = f_batch[0]
            #     ids = f_batch[1]
            #     logit_v = f_batch[3]
            #     # gen = m_batch[2]
            #     female_z = bertflow(embs.to(device), return_loss=False)  # Here z is the sentence embedding
            #     female_z_list.append(female_z)
            #     f_embs.append(embs)
            #     f_ids.append(ids)
            #     f_logits.append(logit_v)
    
        # new_female_z_list = copy.deepcopy(female_z_list)
        # new_male_z_list = copy.deepcopy(male_z_list)
    
        # gender_transfer_count = 0
        # semantic_transfer_count = 0
        # non_conversion_count = 0
        # original_old_word = []
        # old_word = []
        # old_id = []
        # new_word = []
        # new_id = []
        # gender = []
        # logit_vals = []
    
        sum_fem = 0
        count = 0
        for j in range(len(female_z_list)):
            for k in range(len(female_z_list[j][0])):
                sum_fem = sum_fem + female_z_list[j][0][k]
                count+=1
        avg_sum_fem = sum_fem/count
        return avg_sum_fem


    
    #########generate embeddings in gaussian space#################
    def gen_flow_embs_b(self, female_dataloader, male_dataloader, avg_sum_fem, bertflow):
        bertflow.eval()
        
        female_z_list = []
        male_z_list = []
        f_embs = []
        m_embs = []
        f_ids = []
        m_ids = []
        f_logits = []
        m_logits = []
        
        # For each batch of training data...
        bertflow.eval()
        with torch.no_grad():
            for step, f_batch in enumerate(female_dataloader):
                # print(f_batch.shape)
                embs = f_batch[0]
                ids = f_batch[1]
                logit_v = f_batch[3]
                # print(embs.shape)
                female_z = bertflow(embs.to(device), return_loss=False)  # Here z is the sentence embedding
                female_z_list.append(female_z)
                f_embs.append(embs)
                f_ids.append(ids)
                f_logits.append(logit_v)
            for step, m_batch in enumerate(male_dataloader):
                embs = m_batch[0]
                ids = m_batch[1]
                logit_v = m_batch[3]
                # gen = m_batch[2]
                male_z = bertflow(embs.to(device), return_loss=False)  # Here z is the sentence embedding
                male_z_list.append(male_z)
                m_embs.append(embs)
                m_ids.append(ids)
                m_logits.append(logit_v)
    
        new_male_z_list = copy.deepcopy(male_z_list)
        # new_male_z_list = copy.deepcopy(male_z_list)
    
        gender_transfer_count = 0
        semantic_transfer_count = 0
        non_conversion_count = 0
        original_old_word = []
        old_word = []
        old_id = []
        new_word = []
        new_id = []
        gender = []
        logit_vals = []
    
        sum_fem = 0
        count = 0
        for j in range(len(female_z_list)):
            # print("############", j)
            for k in range(len(female_z_list[j][0])):
                sum_fem = sum_fem + female_z_list[j][0][k]
                count+=1
        avg_sum_fem = sum_fem/count
    
        for j in range(len(male_z_list)):
            for k in range(len(male_z_list[j][0])):
            # part1 = torch.tensor([female_z[0][0][15].cpu().detach().numpy() for i in female_z[0][0]])
                # new_male_z_list[j][0][0][k] = female_z_list[j][0][0][k]
                new_male_z_list[j][0][k] = avg_sum_fem
            
                
            eee = bertflow(new_male_z_list[j], reverse=True)
            eee = torch.squeeze(eee)
            ffff = self.decode_emb(eee)
            ffff
            for key,i in enumerate(self.decode_emb(m_embs[j][:,:].to(device))):
                # if f_logits[j][key].item() < -10:
                original_old_word.append(self.vv[m_ids[j][key].item()])
                old_word.append(self.vv[i.item()])
                old_id.append(i.item())
                new_word.append(self.vv[ffff[key].item()])
                new_id.append(ffff[key].item())
                gender.append(1)
                logit_vals.append(m_logits[j][key].item())
        
        return original_old_word, old_word, old_id, new_word, new_id, gender, logit_vals
    
    
    
    
    
    
    
    
    
    def get_instances_b(self, attribute1_list, attribute2_list, chunk_size):
        attribute_pair = f"{attribute1_list[0]}_{attribute2_list[0]}"
        model_path = "outputs/bertflow_model_"+attribute_pair
        bertflow = torch.load(f"{model_path}/bertflow.pth")
        original_old_word = []
        old_word = []
        old_id = []
        new_word = []
        new_id = []
        gender = []
        logit_vals = []
        
        
        for jj in range(1,2):
            for iteration in range(chunk_size, chunk_size+1, chunk_size+1):
                print(iteration)
            
                # with open('../1_viz/attribute_embeddings/embeddings_1000.pkl', 'rb') as f:
                #     embeddings = pickle.load(f)
                # with open('../1_viz/attribute_embeddings/gender_1000.pkl', 'rb') as f:
                #     gen_labels = pickle.load(f)
                # with open('../1_viz/attribute_embeddings/token_ids_1000.pkl', 'rb') as f:
                #     attribute_ids = pickle.load(f)
                with open('outputs/attribute_embeddings_'+attribute_pair+'/embeddings_'+str(iteration)+'.pkl', 'rb') as f:
                    embeddings = pickle.load(f)
                with open('outputs/attribute_embeddings_'+attribute_pair+'/gender_'+str(iteration)+'.pkl', 'rb') as f:
                        gen_labels = pickle.load(f)
                with open('outputs/attribute_embeddings_'+attribute_pair+'/token_ids_'+str(iteration)+'.pkl', 'rb') as f:
                        attribute_ids = pickle.load(f)
                with open('outputs/attribute_embeddings_'+attribute_pair+'/logits_'+str(iteration)+'.pkl', 'rb') as f:
                        logits = pickle.load(f)
        
            
                start_ind = 0
                end_ind = 0
                # ratio = 10000
                ratio = len(attribute_ids)
                
                for i in range(int(len(attribute_ids)/ratio)):
                    print(i)
                    end_ind+=ratio
                    female_embeddings, female_attribute_ids, female_gen_labels, female_logits, male_embeddings, male_attribute_ids, male_gen_labels, male_logits = self.get_examples(embeddings[start_ind:end_ind], gen_labels[start_ind:end_ind], attribute_ids[start_ind:end_ind], logits[start_ind:end_ind], 1, train=False)
                    female_data_pairs_test = TensorDataset(female_embeddings, female_attribute_ids, female_gen_labels, female_logits)
                    male_data_pairs_test = TensorDataset(male_embeddings, male_attribute_ids, male_gen_labels, male_logits)
                    
                    female_dataloader = DataLoader(
                                female_data_pairs_test,  # The training samples.
                                sampler = None, #RandomSampler(female_data_pairs_test), # Select batches randomly
                                batch_size = 20 # Trains with this batch size.
                            )
                    
                    male_dataloader = DataLoader(
                                male_data_pairs_test,  # The training samples.
                                sampler = None, #RandomSampler(male_data_pairs_test), # Select batches randomly
                                batch_size = 20 # Trains with this batch size.
                            )
                    start_ind += ratio
                    # try:
                    # print((female_embeddings.shape), (male_embeddings.shape))
                    avg_sum_fem = self.gen_flow_embs1_b(female_dataloader, bertflow)
                    original_old_word_temp, old_word_temp, old_id_temp, new_word_temp, new_id_temp, gender_temp, logit_vals_temp = self.gen_flow_embs_b(female_dataloader, male_dataloader, avg_sum_fem, bertflow)
                    original_old_word+= original_old_word_temp
                    old_word+= old_word_temp
                    old_id+= old_id_temp
                    new_word+= new_word_temp
                    new_id+= new_id_temp
                    gender+= gender_temp
                    logit_vals+= logit_vals_temp

            
        
        
        dict = {'original_old_word': original_old_word, 'old_word': old_word, 'old_id': old_id, 'new_word': new_word, 'new_id': new_id, 'gender': gender, 'logits': logit_vals}
        df = pd.DataFrame(dict)
        df["count"] = 1 
        df
        
        
        filter = df['original_old_word'].str.contains("##")
        df = df[~filter]
        filter = df['original_old_word'].str.contains("[CLS]")
        df = df[~filter]
        filter = df['original_old_word'].str.contains("[SEP]")
        df = df[~filter]
        filter = df['original_old_word'].str.contains("[PAD]")
        df = df[~filter]
        
        
        df.to_csv('outputs/male_female_full_pairs_'+attribute_pair+'.csv', index=False)