# phase1 train test
python3 /workspace/electra/build_pretraining_dataset.py --corpus-dir ./corpus/phase1/train/ \
                                                        --output-dir ./pretrain_tfrecord2/seq_128/train \
                                                        --vocab-file ./vocab/vocab.txt \
                                                        --no-lower-case \
                                                        --max-seq-length 128 \
                                                        --num-processes 32 \
                                                        --num-out-files 2048

# if you need do_eval step
#python /workspace/electra/build_pretraining_dataset.py --corpus-dir ./corpus/phase1/test/ \
#                                                        --output-dir ./pretrain_tfrecord2/seq_128/test \
#                                                        --vocab-file ./vocab/vocab.txt \
#                                                        --no-lower-case \
#                                                        --max-seq-length 128 \
#                                                        --num-processes 1 \
#                                                         --num-out-files 2048

python /workspace/electra/build_pretraining_dataset.py --corpus-dir ./corpus/phase2/train/ \
                                                        --output-dir ./pretrain_tfrecord2/seq_512/train \
                                                        --vocab-file ./vocab/vocab.txt \
                                                        --no-lower-case \
                                                        --max-seq-length 512 \
                                                        --num-processes 13 \
                                                        --num-out-files 2048
# if you need do_eval step
#python /workspace/electra/build_pretraining_dataset.py --corpus-dir ./corpus/phase2/test/ \
#                                                        --output-dir ./pretrain_tfrecord2/seq_512/test \
#                                                        --vocab-file ./vocab/vocab.txt \
#                                                        --no-lower-case \
#                                                        --max-seq-length 512 \
#                                                        --num-processes 1 \
#                                                        --num-out-files 2048
