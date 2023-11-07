# ZC3 -- Zero-Shot Cross-Language Code Clone Detection 


## Task Definition

Given a code and a collection of candidates as the input, the task is to retrieve codes with the same semantic from a collection of candidates.
Models are evaluated by MAP score. MAP is defined as the mean of average precision scores, which is evaluated for retrieving similar samples given a query. 


## Dataset

We use the CodeJam, AtCoder, CSNCC, and XLCOST datasets dataset for cross-language clone detection. 
And we use the BigCloneBench dataset and the POJ-104 dataset to detect monolingual clones.


### Data Format

At dataset, you can obtain four cross-language clone detection datasets.

Each folder contains two files for a dataset. 
For each file, each line in the uncompressed file represents one function.  One row is illustrated below.

   - **code:** the source code
   - **label:** the number of problem that the source code solves
   - **index:** the index of example

Given a codes file evaluator/test.jsonl:

```bash
{"label": "65", "index": "0", "code": "function0"}
{"label": "65", "index": "1", "code": "function1"}
{"label": "65", "index": "2", "code": "function2"}
{"label": "66", "index": "3", "code": "function3"}
```



### Fine-tune

```shell
export CUDA_VISIBLE_DEVICES=0,1,2
python run.py \
 --output_dir=./saved_models_codes \
 --model_type=roberta \
 --config_name=microsoft/codebert-base \
 --model_name_or_path=microsoft/codebert-base \
 --tokenizer_name=roberta-base \
 --do_train \
 --do_eval \
 --train_data_file /dataset/pj_with_func_train.jsonl \
 --epoch 50 \
 --save_steps=50 \
 --block_size 512 \
 --train_batch_size 32 \
 --eval_batch_size 48 \
 --learning_rate 2e-5 \
 --max_grad_norm 1.0 \
 --evaluate_during_training \
 --seed 123456 
 


### Inference

```shell
export CUDA_VISIBLE_DEVICES=0,1,2
python run.py \
 --output_dir=./saved_models_codes \
 --model_type=roberta \
 --config_name=microsoft/codebert-base \
 --model_name_or_path=microsoft/codebert-base \
 --tokenizer_name=roberta-base \
 --do_eval \
 --train_data_file /dataset/pj_with_func_train.jsonl \
 --query_data_file /dataset/CodeJamData/CodeJamData_py_test.jsonl \
 --candidate_data_file  /dataset/CodeJamData/CodeJamData_java_test.jsonl \
 --epoch 50 \
 --save_steps=50 \
 --block_size 512 \
 --train_batch_size 32 \
 --eval_batch_size 48 \
 --learning_rate 2e-5 \
 --max_grad_norm 1.0 \
 --evaluate_during_training \
 --seed 123456 
 

## AtCoder dataset
 --query_data_file /dataset/AtCoder/AtCoder_py_test.jsonl \
 --candidate_data_file /dataset/AtCoder/AtCoder_java_test.jsonl \


## XLCoST dataset
 --query_data_file /dataset/XLCoST/python_test.jsonl \
 --candidate_data_file /dataset/XLCoST/java_test.jsonl \


## CodeJamData dataset
 --query_data_file /dataset/CodeJamData/CodeJamData_py_test.jsonl \
 --candidate_data_file  /dataset/CodeJamData/CodeJamData_java_test.jsonl \


## CSNCC dataset
 --query_data_file /CSNCC/pj_python_with_func_test.jsonl \
 --candidate_data_file /CSNCC/pj_java_with_func_test.jsonl \





