# Assignment 2
A document indexer, consisting of a corpus reader, 
document processor, tokenizer, a weighted (tf-idf) indexer and a ranked retrieval method

## 1 - Installing dependencies
In order to install the required dependencies, run the following command.

```
pip install -r requirements.txt
```

## 2 - Running the indexer and generating weights
To run the indexer execute the following command. If no filepath is provided the indexer will try to open the file 'datasets/metadata_2020-03-27.csv'

```
cd assignment
python3 indexer.py [filepath]
```

Two files will be generated: outputs/bmc_weights.csv and outputs/tf_idf_weights.csv. These files will be loaded by the ranking entities

## 3 - Vector space ranking with tf-idf weights
To run the vector space ranking execute the following command. If no input_filepath is provided, then the weights will be loaded from 'outputs/tf_idf_weights.csv'

```
python3 vector_space_ranking.py [input_filepath]
```

The results will be generated to 'outputs/vector_space_results.csv'

## 4 - BM25 ranking
To run the bm25 ranking execute the following command. If no input_filepath is providedIf no input_filepath is provided, then the weights will be loaded from 'outputs/bmc_weights.csv'

```
python3 bmc_ranking.py [input_filepath]
```

The results will be generated to 'outputs/bmc_results.csv'