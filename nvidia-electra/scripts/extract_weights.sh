python3  ../transform_weights_from_ckpt_to_h5.py \
                --model_size=base \
                --max_seq_length=512 \
                --amp --xla \
                --vocab_size=32200 \
                --pretrained-checkpoint /workspace/ELECTRA/results/models/base/ckpt-8493
                --output-dir /workspace/ELECTRA/results/models/base  \