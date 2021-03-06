# joint_coref
See log: https://docs.google.com/document/d/1CdORk86nj1NRDXiw7dDjKZpw1IfP4NsgtXhaMTAguH8/edit?usp=sharing


## LEO
python src/all_models/train_model.py --config_path train_config.json --out_dir models/no_head_embeddings

## build_feature and lemma baseline
find . -name '*.DS_Store' -delete
python src/data/make_dataset.py --ecb_path data/ECB+/ECB+/ --output_dir data/prep/ --data_setup 2 --selected_sentences_file data/ECB+/ECBplus_coreference_sentences.csv

python src/features/build_features.py --config_path "build_features_config.json" --output_path "data/feature/"

python3 models/lemma_baseline.py --config_path "lemma_baseline_config.json"


### evaluation (takes some time)

perl scorer/scorer.pl all data/gold/CD_test_entity_mention_based.key_conll  output/baseline/CD_test_entity_mention_based.response_conll >> output/baseline/stat/CD_entity.txt 


perl scorer/scorer.pl all data/gold/wd_test_entity_mention_based.key_conll  output/baseline/wd_test_entity_mention_based.response_conll >> output/baseline/stat/wd_entity.txt 

perl scorer/scorer.pl all data/gold/CD_test_event_mention_based.key_conll  output/baseline/CD_test_event_mention_based.response_conll >> output/baseline/stat/CD_event.txt 

perl scorer/scorer.pl all data/gold/wd_test_event_mention_based.key_conll  output/baseline/wd_test_event_mention_based.response_conll >> output/baseline/stat/wd_event.txt 


## Summarize (coreference score)


python scorer/summarize.py --input_path output/baseline/stat/CD_entity.txt >> output/baseline/stat/s_CD_entity.txt

python scorer/summarize.py --input_path output/baseline/stat/wd_entity.txt >> output/baseline/stat/s_wd_entity.txt

python scorer/summarize.py --input_path output/baseline/stat/CD_event.txt >> output/baseline/stat/s_CD_event.txt

python scorer/summarize.py --input_path output/baseline/stat/wd_event.txt >> output/baseline/stat/s_wd_event.txt


## Notes on setting up environment (Updated March 2021)

Follow the instruction at the deriving repo.

Install all the required package expect PyTorch with
`pip install -r requirements.txt`

We found pytorch 1.2.0 to be a working version. Recommend using Conda for installing a compatible version of pytorch.
```
# CUDA 9.2
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch

# CUDA 10.0
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

# CPU Only
conda install pytorch==1.2.0 torchvision==0.4.0 cpuonly -c pytorch
```