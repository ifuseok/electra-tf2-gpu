from pretrain_config import PretrainingConfig
from modeling import PretrainingModel
import argparse
import collections
import json
import os
from utils import log, heading
import tensorflow as tf
import pickle


def main():
    # Parse essential argumentss
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", default="base", type=str, help="small,small++,base or large",required=True)
    parser.add_argument("--model_name", default="base", type=str, help="small,small++,base or large")
    parser.add_argument("--amp", action='store_true',
                        help="Whether to use fp16.")
    parser.add_argument("--xla", action='store_true',
                        help="Whether to use xla.")
    parser.add_argument("--max_seq_length",default=512, type=int)
    parser.add_argument("--vocab_size",default=32200, type=int)
    parser.add_argument("--output-dir",type=str)
    parser.add_argument("--pretrained-checkpoint",type=str)
    
    args = parser.parse_args()
    config = PretrainingConfig(**args.__dict__)
    
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    if args.amp:
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16", loss_scale="dynamic")
        tf.keras.mixed_precision.experimental.set_policy(policy)
        print('Compute dtype: %s' % policy.compute_dtype)  # Compute dtype: float16
        print('Variable dtype: %s' % policy.variable_dtype)  # Variable dtype: float32

    # Set up model
    model = PretrainingModel(config)


    # Load checkpoint
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), model=model)
    
    checkpoint.restore(args.pretrained_checkpoint)
    
    log(" ** Restored from {} at step {}".format(args.pretrained_checkpoint, int(checkpoint.step) - 1))


    disc_dir = os.path.join(args.output_dir, 'discriminator')
    gen_dir = os.path.join(args.output_dir, 'generator')
    
    heading(" ** Saving discriminator")
    model.discriminator(model.discriminator.dummy_inputs)
    model.discriminator.save_pretrained(disc_dir)

    heading(" ** Saving generator")
    model.generator(model.generator.dummy_inputs)
    model.generator.save_pretrained(gen_dir)
    log(" ** Restored from {} at step {}".format(args.pretrained_checkpoint, int(checkpoint.step) - 1))

if __name__ =="__main__":
    main()