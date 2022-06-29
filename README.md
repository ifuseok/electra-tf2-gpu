# ELECTRA For TensorFlow2(GPU)

- GPU를 활용하여 ELECTRA를 학습하는 과정 중 [Google's implementation](https://github.com/google-research/electra)에서 공개한 코드를 사용할 때 학습 Loss는 떨어져도 fine-tune이 잘 되지 않는 문제가 있었습니다.

- GPU를 활용한 최적화된 코드를 찾던 중[NVIDIA DeeplearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/LanguageModeling/ELECTRA)에서 공개된 Tensorflow2 로 구성된 코드를 사용하는 것이 적합하였습니다.

- NVIDIA 측에서 공개된 코드가 ELECTRA base,large 모델을 학습할 수 있도록 코드가 구성되어 있어 small, small++ 를 학습 가능하도록 추가 수정하였습니다.
- 또한 공개되어 있는 torch로 된 finetune 코드를 tensorflow 전환한 것도 추가하였습니다. torch보다 tensorflow가 익숙하신분들이 사용하시면 될것 같습니다.


### Training

| **Feature** | **ELECTRA** |
|:---------:|:----------:|
|LAMB|Yes|
|Automatic mixed precision (AMP)|Yes|
|XLA|Yes|
|Horovod Multi-GPU|Yes|
|Multi-node|Yes|

- 모델 학습시 지원되는 기능을 보면 위와 같음을 확인할 수 있습니다.
- Multi-GPU 환경에서 학습 가능한 Horovod와 Automatic Mixed Precision(AMP)로 float16으로 연산하여 GPU로도 빠른 학습이 가능합니다.
- 또한, 학습 단계를 1,2단계로 구분하여 96% 데이터는 sequence의 max_length길이를 128로, 4%의 데이터만 max_length길이를 512로 학습하여 빠르게 학습합니다.
- BASE 모델 기준(데이터 30GB) v100 16GB 4대 기준 약 7일이면 학습이 가능합니다.
    1. Phase 1(max length : 128, 데이터 : 96%) :  5 day
    2. Phase 2(max length : 512, 데이터 : 4%) :  2 day



### Pretrain 실행 방법
1. Make TFRecord
- `make_custom_dataset.sh` 내에서
    - `--corpus-dir`에 corpus 데이터 경로
    - `--output-dir`에 tfrecord output 경로
    - `--num-processes`에 분할된 corpus 문서 수 만큼 병렬처리 설정하면됩니다.

```
sh make_custom_dataset.sh
```

2. Modify Config
- `scripts/config/pretrain_config.sh`에서 ELECTRA pretrain 학습에 적용할 옵션들이 정의되어 있으며 수정하여 사용할 수 있습니다. 사용 가능한 GPU 재원에 따라 옵션을 선택하거나 수정하면 됩니다.
- `electra_model` 에서 model type은 `base`,`large`,`small`,`small++` 로 선택 가능합니다.
- 현재 기본 vocab_size는 32200으로 되어 있으며 vocab size를 수정하실 경우 `run_pretraining.py` 와 `pretrain_config.py`의 vocab_size를 수정해야하며, GPU OOM 에러 발생시 `pretrain_config.sh` 내 train_batch_size를 수정하여야 합니다.

| **Model** | **Hidden layers** | **Hidden unit size** | **Parameters** |
|:---------:|:----------:|:---:|:----:|
|ELECTRA_SMALL|12 encoder| 256 | 14M|
|ELECTRA_SMALL++|12 encoder| 512 | 47M|
|ELECTRA_BASE |12 encoder| 768 |110M|
|ELECTRA_LARGE|24 encoder|1024 |335M|
 
3. Run pretrain
- `scripts/run_pretraining_modify.sh` 내에서 TFRecord가 존재하는 경로로 각각 `DATA_DIR_P1`,`DATA_DIR_P2` 를 수정하고 코드가 있는 경로로 `CODEDIR` 수정하여 실행할 수 있습니다.
- `pretrain_config.sh` 내에서 아래와 같이 옵션을 선택하여 학습을 시작할 수 있습니다.
```
$ sh scripts/run_pretraing_custom.sh $(source scripts/configs/pretrain_config.sh && dgx1_4gpu_amp)
```

4. Transform ckpt to h5
- 학습이 완료된 후 `transformers`라이브러리로 weights를 불러오고 싶을 경우 ckpt->h5 , h5->bin 으로 변환하여 pytorch와 tensorflow 에서 사용 가능합니다.(h5를 pytorch 코드로 `from_tf=True`옵션으로 불러와 사용할 경우 GPU memory 이슈 발생할 수 있음)
- `scripts/extract_weights.sh` 내에서 amp 사용 여부에 따라 `--amp`를 추가 및 학습한 vocab,max_seq_length 크기에 맞게 `--vocab_size`와 `--max_seq_length`를 수정하고 `--pretrained-checkpoint`,`--output-dir`를 각각 pretrain으로 얻은 ckpt 경로와 h5를 저장할 경로로 변경하면 됩니다.
- pretrain 학습에 사용했던 config 옵션을 적용하여 실행하면 `--output-dir`내에 `generator`,`discriminator` 경로로 각각 `config.json`과 `tf_model.h5`가 저장됩니다.
```
$ sh scripts/extract_weights.sh
```
- `transformers` 라이브러리 로 모델을 불러올 때  `TFElectraModel`는 `from_pretrained()` method로 h5가 저장된 경로로 path를 설정하면 바로 사용 가능하며 torch 버젼으로 사용하고 싶을 경우 `ElectraModel`의 `from_pretrained(from_tf=True)`로 한번 weights를 불러온 후 `save_pretrained()`로 `pytorch_model.bin`으로 변환하여 저장한 후 다시 불러오면 GPU 메모리 이슈없이 torch로도 사용 가능합니다.


### Requirements
- 학습에 필요한 환경은 NVIDIA 측에서 만들어 놓은 Docker 환경을 그대로 사용면 됩니다. 다만, 이미지를 받아 container를 실행할 경우 `dlloger` 등  추가로 설치가 필요한 패키지가 있습니다. 추가로 올린 Docker hub의 이미지를 사용하시면 별도 패키지 없이 바로 사용 가능합니다.
-   [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
-   [TensorFlow2 20.07-py3 NGC container or later](https://ngc.nvidia.com/registry/nvidia-tensorflow)
-   [추가 패키지 설치 도커 이미지](https://hub.docker.com/repository/docker/ifuseok/nvidia-tf2-py3-electra)




## Reference
- [Google's ELECTRA](https://github.com/google-research/electra)
- [NVIDIA DeeplearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/LanguageModeling/ELECTRA) 
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