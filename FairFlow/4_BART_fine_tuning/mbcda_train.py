from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# from error_correction import gen_text
# import eval_ppl
import pandas as pd
import torch
import pipeline_seq_2_seq
from transformers import AutoTokenizer
import time
import datetime
import os


# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", "--model_name", default="facebook/bart-base", help="generative model type")
parser.add_argument("-p", "--perplexity_test_model", default="distilgpt2", help="model to test for perplexity")
parser.add_argument("-d", "--data_path", default="", help="path to training file - csv")
parser.add_argument("-t", "--train_column", default="text", help="column name for training data")
parser.add_argument("-l", "--label_column", default="flipped_text", help="column name for training labels")
parser.add_argument("-s", "--train_split_ratio", default=0.9, type=float, help="train-test split ratio")
parser.add_argument("-mode", "--mode", default="train", help="train or generate")
parser.add_argument("-c", "--device", default="cuda", help="cpu or cuda")
args = vars(parser.parse_args())
 
# Set up parameters
model_name = args["model_name"]
perplexity_test_model = args["perplexity_test_model"]
data_path = args["data_path"]
train_column = args["train_column"]
label_column = args["label_column"]
train_split_ratio = args["train_split_ratio"]
mode = args["mode"]
device = args["device"]



# import data
df = pd.read_csv(data_path)
# labels = df[label_column].tolist()



# load class objects
tokenizer = AutoTokenizer.from_pretrained(model_name)
adv_model_train = pipeline_seq_2_seq.adv_model_train
# clean_text = gen_text(model_name)
seq_model = adv_model_train(model_name)
# ppl = eval_ppl.perplexity_test(perplexity_test_model)

# process input in chunks that fit in gpu memory
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# time formatting
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


if __name__ == '__main__':
    # ppl_score = ppl.score(labels)
    # print("\n *************** compute perplexity score for dictionary based method ***************")
    # print("perplexity score dictionary", ppl_score)

    # with open('outputs/model_wiki_bart_infill_epoch4_error_logit_0/unfiltered_data/parallel_label_wiki_error_logit_0.txt') as file:
    # # with open('parallel_labels.txt') as file:
    #     # parallel_labels
    #     lines = [line.rstrip() for line in file]
    
    #     df_cleaned = pd.DataFrame(lines)
    #     df_cleaned.reset_index(drop=True)
    #     label_i = list(df_cleaned[0])[:]
    label_i = list(df[label_column])

        
    # # train model
    if mode == "train":

    #     #     #prep data
    #     print("*************** generating parallel data ***************")
    #     label_chunks = list(chunks(labels[:], 300)) #change to 300 for fater processing
    #     label_i = []
    #     label_j = []
    #     gen_text = []
    #     filename1 = "parallel_labels.txt"
    #     try:
    #         os.remove(filename1)
    #     except:
    #         filename1
    #     filename2 = "show_errors.txt"
    #     try:
    #         os.remove(filename2)
    #     except:
    #         filename2
    #     with open(filename1, "a") as f:
    #         for ii, labels_batch in enumerate(label_chunks):
    #             if ii%10==0:
    #                 print(str(ii)+"/"+str(len(label_chunks)))
    #             inter_labels, cleaned_labels = clean_text.generate(labels_batch)
    #             label_i = label_i + cleaned_labels
    #             label_j = label_j + inter_labels
    #             #  print output to file
    #             for output in cleaned_labels:
    #                 output = output.encode('ascii', errors='ignore')
    #                 print(output.decode(), file=f)

    #     with open(filename2, "a") as f:
    #         #  print output to file
    #         for inter_output in label_j:
    #             inter_output = inter_output.encode('ascii', errors='ignore')
    #             print(inter_output.decode(), file=f)
                
        #tokenize data 
        real_input = list(df[:][train_column])[:int(len(label_i)*train_split_ratio)]
        val_input =  list(df[:][train_column])[int(len(label_i)*train_split_ratio):]
        real_gender_labels = list(df[:]["gender"])[:int(len(label_i)*train_split_ratio)]
        val_gender_labels =  list(df[:]["gender"])[int(len(label_i)*train_split_ratio):]
        real_label = label_i[:int(len(label_i)*train_split_ratio)]
        val_label =  label_i[int(len(label_i)*train_split_ratio):]
        print(len(real_input), len(real_label))
        train_dataset= seq_model.data_prep_model_F_training_data(real_input, real_label, real_gender_labels)
        val_dataset= seq_model.data_prep_model_F_training_data(val_input, val_label, val_gender_labels)
        torch.save(train_dataset, "train_dataset.pt")
        torch.save(val_dataset, "val_dataset.pt")
        # train_dataset = torch.load("train_dataset.pt")
        # val_dataset = torch.load("val_dataset.pt")

        
        
        print("*************** training model ***************")
        model = seq_model.train_modelF(train_dataset, val_dataset, 32)
     
        # generate
        print("*************** generating model output ***************")
        filename3 = "MBCDA_output.txt"
        try:
            os.remove(filename3)
        except:
            filename3
        with open(filename3, "a") as f:
            for key1, input_batchh in enumerate(list(chunks(val_input, 100))):
                print(key1)
                with torch.no_grad():
                    inputs = tokenizer.batch_encode_plus(input_batchh, truncation=True, max_length=200, return_tensors="pt", padding=True)
                    input_ids = inputs["input_ids"].to(device)
                    summary_ids = model.generate(input_ids, num_beams=2, do_sample=False, min_length=0, max_length=200)
                    output_text = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    gen_text = gen_text + output_text
                    #  print output to file             
                    for key2, outputt in enumerate(output_text):
                        outputt = outputt.encode('ascii', errors='ignore')
                        print(outputt.decode(), file=f)

    # generate                    
    if mode == "generate":
        #tokenize data      
        val_input =  list(df[train_column][:])
        model.eval()
        model = torch.load("trained_model.pth").eval()
        print("*************** generating model output ***************")
        filename3 = "MBCDA_generate_output_bios_sample.txt"
        try:
            os.remove(filename3)
        except:
            filename3
        with open(filename3, "a") as f:
            for key1, input_batchh in enumerate(list(chunks(val_input, 100))):
                print(key1)
                with torch.no_grad():
                    inputs = tokenizer.batch_encode_plus(input_batchh, truncation=True, max_length=300, return_tensors="pt", padding=True)
                    input_ids = inputs["input_ids"].to(device)
                    summary_ids = model.generate(input_ids, num_beams=2, do_sample=False, min_length=0, max_length=300)
                    output_text = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    # gen_text = gen_text + output_text
                    #  print output to file             
                    for key2, outputt in enumerate(output_text):
                        outputt = outputt.encode('ascii', errors='ignore')
                        print(outputt.decode(), file=f)
        

