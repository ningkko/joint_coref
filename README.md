# joint_coref
See log: https://docs.google.com/document/d/1CdORk86nj1NRDXiw7dDjKZpw1IfP4NsgtXhaMTAguH8/edit?usp=sharing


python3 data/feature/build_feature.py --config_path feature_config.json" --output_path "data/feature/output/"

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
