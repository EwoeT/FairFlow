from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import GenderAttributeTokenizer
import GenderAttributeClassifier
import ExtractAttributeEmbeddings
import pandas as pd
import torch
import os

# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--data_src", help="model type")
parser.add_argument("-m", "--model_name", default='bert-base-uncased', help="model type")
parser.add_argument("-x", "--attribute_list1", nargs="+", help="first attribute")
parser.add_argument("-y", "--attribute_list2",nargs="+", help="second attribute")
parser.add_argument("-l", "--seq_len", default=150, type=int, help="maximum text sequence length")
parser.add_argument("-t", "--threshold", default=3, type=int, help="threshold for attribute extraction")
parser.add_argument("-b", "--batch_size", default=32, type=int, help="batch size")
parser.add_argument("-c", "--chunk_size", default=50000, type=int, help="chunk size")
parser.add_argument("-e", "--epochs", default=4, type=int, help="number of epochs to train attribute classifier")
args = vars(parser.parse_args())


# Set up parameters
data_src = args["data_src"]
model_name = args["model_name"]
attribute_list1 = args["attribute_list1"]
attribute_list2 = args["attribute_list2"]
seq_len = args["seq_len"]
threshold = args["threshold"]
batch_size = args["batch_size"]
chunk_size = args["chunk_size"]
epochs = args["epochs"]

if not os.path.exists("outputs"):
    os.makedirs("outputs")


# tokenize data
with open(data_src) as file:
    lines = [line.rstrip() for line in file]
attribute_df = pd.DataFrame(lines)[0]
attribute_df.reset_index(drop=True)

corpus_tokenizer = GenderAttributeTokenizer.corpus_tokenizer(model_name, attribute_list1, attribute_list2, seq_len)
train_dataset_1, val_dataset_1 = corpus_tokenizer.get_corpus_token_ids(attribute_df[:10000])

# torch.save(train_dataset_1, "outputs/1_train_dataset.pt")
# torch.save(val_dataset_1, "outputs/1_val_dataset.pt")

# train_dataset_1 = torch.load("outputs/1_train_dataset.pt")
# val_dataset_1 = torch.load("outputs/1_val_dataset.pt")
get_attribute_tokens = GenderAttributeTokenizer.get_attribute_tokens(model_name, attribute_list1, attribute_list2, train_dataset_1, val_dataset_1, seq_len)
get_attribute_tokens.generate_tokens()


# train gender classifier
train_attrbitue_classifier = GenderAttributeClassifier.AttributeClassifer(model_name, attribute_list1, attribute_list2, batch_size, epochs)
train_attrbitue_classifier.train_model()


# extract embeddings
extract_embeddings = ExtractAttributeEmbeddings.ExtractAttributeEmbeddings(model_name, attribute_list1, attribute_list2, threshold, chunk_size, batch_size)
extract_embeddings.extract(train_dataset_1)


