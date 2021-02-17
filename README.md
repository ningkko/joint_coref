# joint_coref
See log: https://docs.google.com/document/d/1CdORk86nj1NRDXiw7dDjKZpw1IfP4NsgtXhaMTAguH8/edit?usp=sharing


python3 feature/feature.py --config_path feature_config.json" --output_path "feature/output/"

python3 models/lemma_baseline.py --config_path "lemma_baseline_config.json"

perl scorer/scorer.pl all data/gold_cybulska/CD_test_entity_mention_based.key_conll  statistics/baseline/CD_test_entity_mention_based.response_conll

python scorer/summarize.py --input_path statistics/baseline/CD_eval.txt >> statistics/baseline/CD_eval_summary.txt
