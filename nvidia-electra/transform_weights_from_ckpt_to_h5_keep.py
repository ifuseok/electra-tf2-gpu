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
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_size", default="base", type=str, help="base or large")
    parser.add_argument("--pretrain_tfrecords", type=str)
    parser.add_argument("--phase2", action='store_true')
    parser.add_argument("--fp16_compression", action='store_true')
    parser.add_argument("--amp", action='store_true',
                        help="Whether to use fp16.")
    parser.add_argument("--xla", action='store_true',
                        help="Whether to use xla.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_train_steps", type=int)
    parser.add_argument("--num_warmup_steps", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--max_seq_length", type=int)

    parser.add_argument("--mask_prob", type=float)
    parser.add_argument("--disc_weight", type=float)
    parser.add_argument("--generator_hidden_size", type=float)

    parser.add_argument("--log_freq", type=int, default=10, help="Training metrics logging frequency")
    parser.add_argument("--save_checkpoints_steps", type=int)
    parser.add_argument("--keep_checkpoint_max", type=int)
    parser.add_argument("--restore_checkpoint", default=None, type=str)
    parser.add_argument("--load_weights", action='store_true')
    parser.add_argument("--weights_dir")

    parser.add_argument("--optimizer", default="adam", type=str, help="adam or lamb")
    parser.add_argument("--skip_adaptive", action='store_true', help="Whether to apply adaptive LR on LayerNorm and biases")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of Gradient Accumulation steps")
    parser.add_argument("--lr_decay_power", type=float, default=0.5, help="LR decay power")
    parser.add_argument("--opt_beta_1", type=float, default=0.878, help="Optimizer beta1")
    parser.add_argument("--opt_beta_2", type=float, default=0.974, help="Optimizer beta2")
    parser.add_argument("--end_lr", type=float, default=0.0, help="Ending LR")
    parser.add_argument("--log_dir", type=str, default=None, help="Path to store logs")
    parser.add_argument("--results_dir", type=str, default=None, help="Path to store all model results")
    parser.add_argument("--skip_checkpoint", action='store_true', default=False, help="Path to store logs")
    parser.add_argument('--json-summary', type=str, default=None,
                        help='If provided, the json summary will be written to the specified file.')
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
    
    #manager = tf.train.CheckpointManager(checkpoint, args.pretrained_checkpoint,max_to_keep=5)
    #checkpoint.restore(manager.latest_checkpoint)

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

    #with open(disc_dir+"/pretrain_config.pkl","wb") as f:
    #    pickle.dump(config,f) 

if __name__ =="__main__":
    main()