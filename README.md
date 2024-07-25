## Requirements
1. torch
2. transformers 4.39.0

## 1. Extract attribute words relating to demograhic axis
- Example:
```
!python src/attribute_extraction/viz.py -d '../../DIIN/data/intrinsic_word_level_data/enwiki-20230320-pages-articles4.txt' -m 'bert-base-uncased' -x "she" -y "he" -l 150 -t 5 -b 32 -c 30000
```
<br/>
Args: <br/>
-parser.add_argument("-d", "--data_src", help="model type")
-parser.add_argument("-m", "--model_name", default='bert-base-uncased', help="model type") <br/>
-parser.add_argument("-x", "--attribute_list1", nargs="+", help="first attribute") <br/>
-parser.add_argument("-y", "--attribute_list2",nargs="+", help="second attribute") <br/>
-parser.add_argument("-l", "--seq_len", default=150, type=int, help="maximum text sequence length") <br/>
-parser.add_argument("-t", "--threshold", default=3, type=int, help="threshold for attribute extraction") <br/>
-parser.add_argument("-b", "--batch_size", default=32, type=int, help="batch size") <br/>
-parser.add_argument("-c", "--chunk_size", default=50000, type=int, help="chunk size") <br/>
-parser.add_argument("-e", "--epochs", default=4, type=int, help="number of epochs to train attribute classifier") <br/>



## 2. Generate counterfactual pairs
- Example:
```
!python src/DIIN_counterfactual_generation/DIIN_generate_counterfactuals.py -m 'bert-base-uncased' -x "she" -y "he" -n 10 -r 0.98 -c 30000
```
<br/>
Args: <br/>
-parser.add_argument("-m", "--model_name", default='bert-base-uncased', help="model type") <br/>
-parser.add_argument("-x", "--attribute_list1", nargs="+", help="first attribute") <br/>
-parser.add_argument("-y", "--attribute_list2",nargs="+", help="second attribute") <br/>
-parser.add_argument("-n", "--n_factors", default=10, type=int, help="number of factors for invertible flow network") <br/>
-parser.add_argument("-r", "--rho", default=0.98, type=float, help="correlation factor (how similar embeddings of embedding pairs should be in the common dimensions)") <br/>
-parser.add_argument("-b", "--batch_size", default=32, type=int, help="batch size") <br/>
-parser.add_argument("-c", "--chunk_size", default=30000, type=int, help="chunk size") <br/>




## 3. Error correction
- Example:
```
!python src/error_correction/mbcda.py -m "facebook/bart-base" -d "input_text.txt" -s 0.9 -c "cuda" -mode "train" -l 50 -b 2
```
<br/>
Args: <br/>
-parser.add_argument("-m", "--model_name", default="facebook/bart-base", help="generative model type") <br/>
-parser.add_argument("-d", "--data_path", default="", help="path to training file - csv") <br/>
-parser.add_argument("-s", "--train_split_ratio", default=0.9, type=float, help="train-test split ratio") <br/>
-parser.add_argument("-b", "--num_beams", default=2, type=int, help="number of beams for beam search") <br/>
-parser.add_argument("-mode", "--mode", default="train", help="train or generate") <br/>
-parser.add_argument("-c", "--device", default="cuda", help="cpu or cuda") <br/>
-parser.add_argument("-l", "--max_length", default=200, type=int, help="maximum sentence length") <br/>

