from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# from error_correction import gen_text
# import eval_ppl
import pandas as pd
import torch
# import pipeline_seq_2_seq
from transformers import AutoTokenizer, BartForConditionalGeneration
import time
import datetime
import os


# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", "--model_name", default="facebook/bart-base", help="generative model type")
parser.add_argument("-p", "--perplexity_test_model", default="distilgpt2", help="model to test for perplexity")
parser.add_argument("-model_path", "--model_path", default="", help="path to pretrained model")
parser.add_argument("-data_path", "--data_path", default="", help="path to training file - txt")
parser.add_argument("-num_beams", "--num_beams", default=2, type=int, help="number of beams for text generation")
parser.add_argument("-mode", "--mode", default="train", help="train or generate")
parser.add_argument("-device", "--device", default="cuda", help="cpu or cuda")
args = vars(parser.parse_args())
 
# Set up parameters
model_name = args["model_name"]
perplexity_test_model = args["perplexity_test_model"]
data_path = args["data_path"]
model_path = args["model_path"]
mode = args["mode"]
num_beams = args["num_beams"]
device = args["device"]



# import data
df = pd.read_fwf(data_path, header=None)[0]
# df = pd.read_csv(data_path)
labels = df.tolist()



# load class objects
tokenizer = AutoTokenizer.from_pretrained(model_name)

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
    # generate                    
    if mode == "generate":     
        val_input =  list(df)
        # model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device).eval()
        # model = BartForConditionalGeneration.from_pretrained(model_path).to(device).eval()
        model = torch.load(model_path).eval()

        # generate
        print("*************** generating model output ***************")
        filename3 = "output.txt"
        try:
            os.remove(filename3)
        except:
            filename3
        with open(filename3, "a") as f:
            for key1, input_batchh in enumerate(list(chunks(val_input, 30))):
                print(key1)
                with torch.no_grad():
                    inputs = tokenizer.batch_encode_plus(input_batchh, truncation=True, max_length=300, return_tensors="pt", padding=True)
                    input_ids = inputs["input_ids"].to(device)
                    summary_ids = model.generate(input_ids, num_beams=num_beams, do_sample=False, min_length=0, max_length=300)
                    output_text = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    if key1<1:
                        print(output_text)
            
                    for key2, outputt in enumerate(output_text):
                        outputt = outputt.encode('ascii', errors='ignore')
                        print(outputt.decode(), file=f)

                
