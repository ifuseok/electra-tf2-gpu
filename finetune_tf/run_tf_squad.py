import argparse
import json
import glob
import logging
import os
import random
import re
import timeit
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Reduce the amount of console output from TF
import tensorflow as tf
# Check GPU
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices)>1:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import numpy as np
from fastprogress.fastprogress import master_bar, progress_bar
from tqdm import tqdm
from attrdict import AttrDict
from warnings import filterwarnings
filterwarnings("ignore")


from transformers import (
    squad_convert_examples_to_features
)

# Check GPU
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices)>1:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from src import (
    eval_during_train,
    CONFIG_CLASSES,
    TOKENIZER_CLASSES,
    MODEL_FOR_QUESTION_ANSWERING,
    init_logger,
    set_seed
)


from optimizer import AdamWeightDecay,create_optimizer,GradientAccumulator

from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor

logger = logging.getLogger(__name__)




@tf.function
def train_one_step(args,model,optimizer,batch,accumulator,first_step,take_step,global_step,clip_norm=1.0):
    # Forward and Backward pass
    with tf.GradientTape() as tape:
        total_loss = tf.reduce_mean(model(batch).loss)
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


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    train_num_elements = len(list(train_dataset))#tf.data.experimental.cardinality(train_dataset).numpy()
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (train_num_elements // args.gradient_accumulation_steps) + 1
    else:
        t_total = train_num_elements // args.gradient_accumulation_steps * args.num_train_epochs

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
    logger.info("  Num examples = %d", train_num_elements)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Train batch size per GPU = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to global_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (train_num_elements // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (train_num_elements // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    mb = master_bar(range(int(args.num_train_epochs)))
    # Added here for reproducibility
    set_seed(args)
    for epoch in mb:
        epoch_iterator = progress_bar(train_dataset, parent=mb,total=train_num_elements)
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1

            X,y = batch
            
            inputs_dict = {
                    "input_ids": X["input_ids"],
                    "attention_mask": X["attention_mask"],
                    "token_type_ids": X["token_type_ids"],
                    "start_positions": y["start_positions"],
                    "end_positions": y["end_positions"],
            }
            if args.model_type in ["xlm", "roberta", "distilbert", "distilkobert", "xlm-roberta"]:
                del inputs_dict["token_type_ids"]

            if args.model_type in ["xlnet", "xlm"]:
                inputs_dict.update({"cls_index": y["cls_index"], "p_mask": y["p_mask"]})
                if args.version_2_with_negative:
                    inputs_dict.update({"is_impossible": y["is_impossible"]})
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs_dict.update(
                        {"langs":(tf.ones(inputs_dict["input_ids"],dtype=tf.int64) * args.lang_id)}
                    )

            loss,global_step = train_one_step(args,model,optimizer,inputs_dict,accumulator,step==1,
                    take_step = step % args.gradient_accumulation_steps == 0,
                    global_step = global_step,
                    clip_norm=args.max_grad_norm)

            tr_loss += loss.numpy()
            # Log metrics
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # Only evaluate when single GPU otherwise metrics may not average well
                if args.evaluate_during_training:
                    results = evaluate(args, model, tokenizer, global_step=global_step)
                    for key in sorted(results.keys()):
                        logger.info("  %s = %s", key, str(results[key]))
                logging_loss = tr_loss

            # Save model checkpoint
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                with open(os.path.join(output_dir, "training_args.json"),"w") as f:
                    json.dump(args,f)
                logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        mb.write("Epoch {} done".format(epoch+1))

        if args.max_steps > 0 and global_step > args.max_steps:
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, global_step=None):
    eval_dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    eval_num_elements = len(list(eval_dataset))#tf.data.experimental.cardinality(eval_dataset).numpy()

    # Eval!
    logger.info("***** Running evaluation {} *****".format(global_step))
    logger.info("  Num examples = %d", eval_num_elements)
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()
    eval_step = 0
    for batch in progress_bar(eval_dataset,total=eval_num_elements):
        X,y = batch

        inputs_dict = {
                    "input_ids": X["input_ids"],
                    "attention_mask": X["attention_mask"],
                    "token_type_ids": X["token_type_ids"],
                    "start_positions": y["start_positions"],
                    "end_positions": y["end_positions"],
            }


        if args.model_type in ["xlm", "roberta", "distilbert", "distilkobert", "xlm-roberta"]:
            del inputs_dict["token_type_ids"]

        #example_indices = y["start_positions"]
        example_indices = np.arange(start=(eval_step * args.eval_batch_size)
        ,stop=((eval_step+1) * args.eval_batch_size))
        output = model(inputs_dict)
        eval_step += 1

        for i, example_index in enumerate(example_indices[:len(inputs_dict["input_ids"])]):
            eval_feature = features[example_index]
            unique_id = int(eval_feature.unique_id)

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output.start_logits.numpy()[i].tolist()
                start_top_index = output.start_top_index.numpy()[i].tolist()
                end_logits = output.end_logits.numpy()[i].tolist()
                end_top_index = output.end_top_index.numpy()[i].tolist()
                cls_logits = output.cls_logits.numpy()[i].tolist()

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output.start_logits.numpy()[i].tolist(), output.end_logits.numpy()[i].tolist()
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / eval_num_elements)

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(global_step))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(global_step))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(global_step))
    else:
        output_null_log_odds_file = None

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    # Write the result
    # Write the evaluation result on file
    output_dir = os.path.join(args.output_dir, 'eval')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir, "eval_result_{}_{}.txt".format(list(filter(None, args.model_name_or_path.split("/"))).pop(),
                                                                               global_step))

    with open(output_eval_file, "w", encoding='utf-8') as f:
        official_eval_results = eval_during_train(args, step=global_step)
        results.update(official_eval_results)

    return results


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )
    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        
        dataset = tf.data.experimental.load(cached_features_file)
        with open(os.path.join(cached_features_file,"examples.pkl"),"rb") as f:
            examples = pickle.load(f)
        with open(os.path.join(cached_features_file,"features.pkl"),"rb") as f:
            features = pickle.load(f)
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
            if evaluate:
                examples = processor.get_dev_examples(os.path.join(args.data_dir, args.task),
                                                      filename=args.predict_file)
            else:
                examples = processor.get_train_examples(os.path.join(args.data_dir, args.task),
                                                        filename=args.train_file)

        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset=None,
            threads=args.threads,
        )
        dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="tf",
            threads=args.threads,
        )

        logger.info("Saving features into cached file %s", cached_features_file)
        tf.data.experimental.save(dataset,cached_features_file)
        with open(os.path.join(cached_features_file,"examples.pkl"),"wb") as f:
            pickle.dump(examples,f)
        with open(os.path.join(cached_features_file,"features.pkl"),"wb") as f:
            pickle.dump(features,f)
    if evaluate==False:
        dataset = dataset.batch(args.train_batch_size).shuffle(len(features),seed=args.seed)
    else:
        dataset = dataset.batch(args.eval_batch_size)
    if output_examples:
        return dataset, examples, features
    return dataset


def main(cli_args):
    # Read from config file and make args
    with open(os.path.join(cli_args.config_dir, cli_args.task, cli_args.config_file)) as f:
        args = AttrDict(json.load(f))
    logger.info("Training/evaluation parameters {}".format(args))

    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    init_logger()
    set_seed(args)

    logging.getLogger("transformers.data.metrics.squad_metrics").setLevel(logging.WARN)  # Reduce model loading logs

    # Load pretrained model and tokenizer
    config = CONFIG_CLASSES[args.model_type].from_pretrained(
        args.model_name_or_path,
    )
    tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
    )
    try:
        model = MODEL_FOR_QUESTION_ANSWERING[args.model_type].from_pretrained(
            args.model_name_or_path,
            config=config,
        )
    except:
        model = MODEL_FOR_QUESTION_ANSWERING[args.model_type].from_pretrained(
            args.model_name_or_path,
            config=config,from_pt=True
        )

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval:
        checkpoints = list(os.path.dirname(c) for c in
                           sorted(glob.glob(args.output_dir + "/**/" + "tf_model.h5", recursive=True),
                                  key=lambda path_with_step: list(map(int, re.findall(r"\d+", path_with_step)))[-1]))
        if not args.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        else:
            logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce model loading logs
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1]
            model = MODEL_FOR_QUESTION_ANSWERING[args.model_type].from_pretrained(checkpoint)
            result = evaluate(args, model, tokenizer, global_step=global_step)
            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

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


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--task", type=str, required=True)
    cli_parser.add_argument("--config_dir", type=str, default="config")
    cli_parser.add_argument("--config_file", type=str, required=True)

    cli_args = cli_parser.parse_args()

    main(cli_args)
