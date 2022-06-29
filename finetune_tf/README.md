# Finetuning (Benchmark on subtask)

- Jangwon Park님이 제작한 [KoELECTRA Finetune Code](https://github.com/monologg/KoELECTRA/tree/master/finetune)를 fork하여 Torch로 된 코드를 Tensorflow egar mode 로 수정 한 코드입니다.
-`tansformers v4`기준으로 정상동작 되는 것을 확인했습니다.

## Requirements

```python
tensorflow==2.8.0
transformers==4.19.2
seqeval
fastprogress
attrdict
```

## How to Run

```bash
$ python3 run_tf_seq_cls.py --task {$TASK_NAME} --config_file {$CONFIG_FILE}
```

```bash
$ python3 run_tf_seq_cls.py --task nsmc --config_file koelectra-base.json
$ python3 run_tf_seq_cls.py --task kornli --config_file koelectra-base.json
$ python3 run_tf_seq_cls.py --task paws --config_file koelectra-base.json
$ python3 run_tf_seq_cls.py --task question-pair --config_file koelectra-base-v2.json
$ python3 run_tf_seq_cls.py --task korsts --config_file koelectra-small-v2.json
$ python3 run_tf_ner.py --task naver-ner --config_file koelectra-small.json
$ python3 run_tf_squad.py --task korquad --config_file xlm-roberta.json
```


## Reference
- [Transformers Examples](https://github.com/huggingface/transformers/blob/master/examples/README.md)
- [NSMC](https://github.com/e9t/nsmc)
- [Naver NER Dataset](https://github.com/naver/nlp-challenge)
- [PAWS](https://github.com/google-research-datasets/paws)
- [KorNLI/KorSTS](https://github.com/kakaobrain/KorNLUDatasets)
- [Question Pair](https://github.com/songys/Question_pair)
- [KorQuad](https://korquad.github.io/category/1.0_KOR.html)
- [Korean Hate Speech](https://github.com/kocohub/korean-hate-speech)
- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [HanBERT](https://github.com/tbai2019/HanBert-54k-N)
- [HanBert Transformers](https://github.com/monologg/HanBert-Transformers)
- [KoELECTRA](https://github.com/monologg/KoELECTRA)