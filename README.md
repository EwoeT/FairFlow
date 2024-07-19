## 2. Extract attribute words relating to demograhic axis
- Args: <br/>
-parser.add_argument("-d", "--data_src", help="model type")
-parser.add_argument("-m", "--model_name", default='bert-base-uncased', help="model type")
-parser.add_argument("-x", "--attribute_list1", nargs="+", help="first attribute")
-parser.add_argument("-y", "--attribute_list2",nargs="+", help="second attribute")
-parser.add_argument("-l", "--seq_len", default=150, type=int, help="maximum text sequence length")
-parser.add_argument("-t", "--threshold", default=3, type=int, help="threshold for attribute extraction")
-parser.add_argument("-b", "--batch_size", default=32, type=int, help="batch size")
-parser.add_argument("-c", "--chunk_size", default=50000, type=int, help="chunk size")
-parser.add_argument("-e", "--epochs", default=4, type=int, help="number of epochs to train attribute classifier")

- Example:
```
!python src/attribute_extraction/viz.py -d '../../DIIN/data/intrinsic_word_level_data/enwiki-20230320-pages-articles4.txt' -m 'bert-base-uncased' -x "she" -y "he" -l 150 -t 5 -b 32 -c 30000
```


## 2. Extract attribute words relating to demograhic axis
- Args: <br/>
-parser.add_argument("-m", "--model_name", default='bert-base-uncased', help="model type")
-parser.add_argument("-x", "--attribute_list1", nargs="+", help="first attribute")
-parser.add_argument("-y", "--attribute_list2",nargs="+", help="second attribute")
-parser.add_argument("-n", "--n_factors", default=10, type=int, help="number of factors for invertible flow network")
-parser.add_argument("-r", "--rho", default=0.98, type=float, help="correlation factor (how similar embeddings of embedding pairs should be in the common dimensions)")
-parser.add_argument("-b", "--batch_size", default=32, type=int, help="batch size")
-parser.add_argument("-c", "--chunk_size", default=30000, type=int, help="chunk size")

- Example:
```
!python src/DIIN_counterfactual_generation/DIIN_generate_counterfactuals.py -m 'bert-base-uncased' -x "she" -y "he" -n 10 -r 0.98 -c 30000
```
