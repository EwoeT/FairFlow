from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import TrainDIIN
import invert

# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", "--model_name", default='bert-base-uncased', help="model type")
parser.add_argument("-x", "--attribute_list1", nargs="+", help="first attribute")
parser.add_argument("-y", "--attribute_list2",nargs="+", help="second attribute")
parser.add_argument("-n", "--n_factors", default=10, type=int, help="number of factors for invertible flow network")
parser.add_argument("-r", "--rho", default=0.98, type=float, help="correlation factor (how similar embeddings of embedding pairs should be in the common dimensions)")
parser.add_argument("-b", "--batch_size", default=32, type=int, help="batch size")
parser.add_argument("-c", "--chunk_size", default=30000, type=int, help="chunk size")
args = vars(parser.parse_args())


# Set up parameters
model_name = args["model_name"]
attribute_list1 = args["attribute_list1"]
attribute_list2 = args["attribute_list2"]
n_factors = args["n_factors"]
batch_size = args["batch_size"]
chunk_size = args["chunk_size"]
rho = args["rho"]


train_diin = TrainDIIN.TrainDIIN()


# n_factors= 10
in_channel = 768
n_flow = 6
hidden_depth = 2
hidden_dim = 100
# rho = 0.98



# chunk_size = 30000

train_diin.train_model(attribute_list1, attribute_list2, n_factors, in_channel, n_flow, hidden_depth, hidden_dim, rho, batch_size, chunk_size)

get_inverted_samples = invert.invert(model_name)

get_inverted_samples.get_instances_a(attribute_list1, attribute_list2, chunk_size)

get_inverted_samples.get_instances_b(attribute_list1, attribute_list2, chunk_size)