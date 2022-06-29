import argparse
import json
import logging
import os
import glob
import re
import sys
from pathlib import Path
from typing import Optional
from attrdict import AttrDict
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Reduce the amount of console output from TF
import tensorflow as tf
from fastprogress.fastprogress import master_bar, progress_bar

# Check GPU
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices)>1:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


import numpy as np


from src import (
    CONFIG_CLASSES,
    TOKENIZER_CLASSES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION,
    init_logger,
    set_seed,
    compute_metrics
)

from optimizer import AdamWeightDecay,create_optimizer,GradientAccumulator

from processor import seq_cls_load_and_cache_examples as load_and_cache_examples
from processor import seq_cls_tasks_num_labels as tasks_num_labels
from processor import seq_cls_processors as processors
from processor import seq_cls_output_modes as output_modes


logger = logging.getLogger(__name__)



@tf.function
def train_one_step(args,model,loss_fn,optimizer,batch,accumulator,first_step,take_step,global_step,clip_norm=1.0):
    # Forward and Backward pass
    x,y = batch
    with tf.GradientTape() as tape:
        logits = model(x).logits
        total_loss = loss_fn(y, logits)
        unscaled_loss = tf.stop_gradient(total_loss)
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    
    #Accumulate gradients
    accumulator(gradients)

    if first_step or take_step:
        #All reduce and Clip the accumulated gradients
        #allreduced_accumulated_gradients = [None if g is None else hvd.allreduce(g / tf.cast(config.gradient_accumulation_steps, g.dtype),
        #                        compression=Compression.fp16 if config.amp and config.fp16_compression else Compression.none)
        #                        for g in accumulator.gradients]
        allreduced_accumulated_gradients = [None if g is None else g for g in accumulator.gradients]
        (clipped_accumulated_gradients, _) = tf.clip_by_global_norm(allreduced_accumulated_gradients, clip_norm=clip_norm)
        #Weight update
        optimizer.apply_gradients(zip(clipped_accumulated_gradients, model.trainable_variables))
        accumulator.reset()
        global_step += 1
    return unscaled_loss,global_step

def train(args,
          model,
          train_dataset,
          dev_dataset=None,
          test_dataset=None):
    train_num_elements = tf.data.experimental.cardinality(train_dataset).numpy()

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (train_num_elements // args.gradient_accumulation_steps) + 1
    else:
        t_total = train_num_elements // args.gradient_accumulation_steps * args.num_train_epochs

    #no_decay = ["layer_norm", "bias", "LayerNorm"]
    if output_modes[args.task] == "classification":
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
    elif output_modes[args.task] == "regression":
        loss_fn = tf.keras.losses.MeanSquaredError()

    accumulator = GradientAccumulator()

    optimizer = create_optimizer(init_lr=args.learning_rate, 
                    num_train_steps=t_total,
                    num_warmup_steps=int(t_total * args.warmup_proportion),
                    weight_decay_rate=args.weight_decay,
                    clip_norm=args.max_grad_norm,
                    epsilon=args.adam_epsilon,
                    #beta_1=args.adam_beta1,
                    #beta_2=args.adam_beta2
                    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)
    accumulator.reset()
    
    global_step = 0
    tr_loss = 0.0

    mb = master_bar(range(int(args.num_train_epochs)))
    for epoch in mb:
        epoch_iterator = progress_bar(train_dataset, parent=mb)
        for step, batch in enumerate(epoch_iterator):
            #if args.model_type not in ["distilkobert", "xlm-roberta"]:
            #    batch[0].pop("token_type_ids") # Distilkobert, XLM-Roberta don't use segment_ids
            loss,global_step = train_one_step(args,model,loss_fn,optimizer,batch,accumulator,step==1,
                    take_step = step % args.gradient_accumulation_steps == 0,
                    global_step = global_step,
                    clip_norm=args.max_grad_norm)

            tr_loss += loss
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                if args.evaluate_test_during_training:
                    evaluate(args, model, test_dataset, "test", global_step)
                else:
                    evaluate(args, model, dev_dataset, "dev", global_step)

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                # Save Model Checkpoint
                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model.save_pretrained(output_dir)

                logger.info("Saving model checkpoint to {}".format(output_dir))
                with open(os.path.join(output_dir, "training_args.json"),"w") as f:
                    json.dump(args,f)
        if args.max_steps > 0 and global_step > args.max_steps:
            break

        mb.write("Epoch {} done".format(epoch + 1))
        if args.max_steps > 0 and global_step > args.max_steps:
            break

    return global_step, tr_loss.numpy() / global_step


def evaluate(args,model,eval_dataset,mode,global_step=None):
    # Eval!
    results = {}

    if global_step != None:
        logger.info("***** Running evaluation on {} dataset ({} step) *****".format(mode, global_step))
    else:
        logger.info("***** Running evaluation on {} dataset *****".format(mode))
    eval_num_elements = tf.data.experimental.cardinality(eval_dataset).numpy()
    logger.info("  Num examples Step = {}".format(eval_num_elements))
    logger.info("  Eval Batch size = {}".format(args.eval_batch_size))

    if output_modes[args.task] == "classification":
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
    elif output_modes[args.task] == "regression":
        loss_fn = tf.keras.losses.MeanSquaredError()
        
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in progress_bar(eval_dataset):
        #if args.model_type not in ["distilkobert", "xlm-roberta"]:
        #    batch[0].pop("token_type_ids") # Distilkobert, XLM-Roberta don't use segment_ids
        
        x,labels = batch
        
        logits = model(x).logits
        tmp_eval_loss = loss_fn(labels, logits).numpy()
        eval_loss += tmp_eval_loss
        
        nb_eval_steps += 1
        if preds is None:
            preds = logits.numpy()
            out_label_ids = labels.numpy()
        else:
            preds = np.append(preds,logits.numpy(),axis=0)
            out_label_ids = np.append(out_label_ids,labels.numpy(),axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if output_modes[args.task] == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_modes[args.task] == "regression":
        preds = np.squeeze(preds)

    result = compute_metrics(args.task, out_label_ids, preds)
    results.update(result)

    output_dir = os.path.join(args.output_dir,mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir, "{}-{}.txt".format(mode, global_step) if global_step else "{}.txt".format(mode))
    with open(output_eval_file, "w") as f_w:
        logger.info("***** Eval results on {} dataset *****".format(mode))
        for key in sorted(results.keys()):
            logger.info("  {} = {}".format(key, str(results[key])))
            f_w.write("  {} = {}\n".format(key, str(results[key])))
    return results


def main(cli_args):
    # Read from config file and make args
    with open(os.path.join(cli_args.config_dir, cli_args.task, cli_args.config_file)) as f:
        args = AttrDict(json.load(f))

    logger.info("Training/evalue paramters {}".format(args))

    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

    init_logger()
    set_seed(args)
    processor = processors[args.task](args)
    labels = processor.get_labels()
    if output_modes[args.task] == "regression":
        config = CONFIG_CLASSES[args.model_type].from_pretrained(
            args.model_name_or_path,
            num_labels=tasks_num_labels[args.task]
        )
    else:
        config = CONFIG_CLASSES[args.model_type].from_pretrained(
            args.model_name_or_path,
            num_labels=tasks_num_labels[args.task],
            id2label={str(i): label for i, label in enumerate(labels)},
            label2id={label: i for i, label in enumerate(labels)},
        )

    tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
    try:
        model = MODEL_FOR_SEQUENCE_CLASSIFICATION[args.model_type].from_pretrained(
            args.model_name_or_path,
            config=config
        )
    except:
        model = MODEL_FOR_SEQUENCE_CLASSIFICATION[args.model_type].from_pretrained(
            args.model_name_or_path,
            config=config,from_pt=True
        )
    
    # Load dataset
    train_dataset = load_and_cache_examples(args, tokenizer, mode="train") if args.train_file else None
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev") if args.dev_file else None
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test") if args.test_file else None

    if dev_dataset == None:
        args.evaluate_test_during_training = True  # If there is no dev dataset, only use testset

    if args.do_train:
        global_step, tr_loss = train(args, model, train_dataset, dev_dataset, test_dataset)
        logger.info(" global_step = {}, average loss = {}".format(global_step, tr_loss))
    
    results = {}
    if args.do_eval:
        checkpoints = list(os.path.dirname(c) for c in
                           sorted(glob.glob(args.output_dir + "/**/" + "tf_model.h5", recursive=True),
                                  key=lambda path_with_step: list(map(int, re.findall(r"\d+", path_with_step)))[-1]))
        
        if not args.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        else:
            logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce logging
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            model = MODEL_FOR_SEQUENCE_CLASSIFICATION[args.model_type].from_pretrained(checkpoint)
            result = evaluate(args, model, test_dataset, mode="test", global_step=global_step)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)
            print(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as f_w:
            if len(checkpoints) > 1:
                for key in sorted(results.keys(), key=lambda key_with_step: (
                        "".join(re.findall(r'[^_]+_', key_with_step)),
                        int(re.findall(r"_\d+", key_with_step)[-1][1:])
                )):
                    f_w.write("{} = {}\n".format(key, str(results[key])))
            else:
                for key in sorted(results.keys()):
                    f_w.write("{} = {}\n".format(key, str(results[key])))

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--task", type=str, required=True)
    cli_parser.add_argument("--config_dir", type=str, default="config")
    cli_parser.add_argument("--config_file", type=str, required=True)

    cli_args = cli_parser.parse_args()

    main(cli_args)