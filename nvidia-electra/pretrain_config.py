import os

class PretrainingConfig(object):
    """Defines pre-training hyperparameters."""

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.seed = 42

        self.debug = False  # debug mode for quickly running things
        self.do_train = True  # pre-train ELECTRA
        self.do_eval = False  # evaluate generator/discriminator on unlabeled data
        self.phase2 = False

        # amp
        self.amp = True
        self.xla = True
        self.fp16_compression = False

        # optimizer type
        self.optimizer = 'adam'
        self.gradient_accumulation_steps = 1

        # lamb whitelisting for LN and biases
        self.skip_adaptive = False

        # loss functions
        self.electra_objective = True  # if False, use the BERT objective instead
        self.gen_weight = 1.0  # masked language modeling / generator loss
        self.disc_weight = 50.0  # discriminator loss
        self.mask_prob = 0.15  # percent of input tokens to mask out / replace

        # optimization
        self.learning_rate = 5e-4
        self.lr_decay_power = 0.5
        self.weight_decay_rate = 0.01
        self.num_warmup_steps = 10000
        self.opt_beta_1 = 0.878
        self.opt_beta_2 = 0.974
        self.end_lr = 0.0

        # training settings
        self.log_freq = 10
        self.skip_checkpoint = False
        self.save_checkpoints_steps = 1000
        self.num_train_steps = 1000000
        self.num_eval_steps = 100
        self.keep_checkpoint_max = 5  # maximum number of recent checkpoint files to keep;  change to 0 or None to keep all checkpoints
        self.restore_checkpoint = None
        self.load_weights = False

        # model settings
        self.model_size = "base"  # one of "small", "base", or "large"
        # override the default transformer hparams for the provided model size; see
        # modeling.BertConfig for the possible hparams and util.training_utils for
        # the defaults
        self.model_hparam_overrides = (
            kwargs["model_hparam_overrides"]
            if "model_hparam_overrides" in kwargs else {})
        self.embedding_size = None  # bert hidden size by default
        self.vocab_size = 32200  # number of tokens in the vocabulary
        self.do_lower_case = False  # lowercase the input?

        # generator settings
        self.uniform_generator = False  # generator is uniform at random
        self.shared_embeddings = True  # share generator/discriminator token embeddings?
        # self.untied_generator = True  # tie all generator/discriminator weights?
        self.generator_layers = 1.0  # frac of discriminator layers for generator
        self.generator_hidden_size = 0.25  # frac of discrim hidden size for gen
        self.disallow_correct = False  # force the generator to sample incorrect
        # tokens (so 15% of tokens are always
        # fake)
        self.temperature = 1.0  # temperature for sampling from generator

        # batch sizes
        self.max_seq_length = 128
        self.train_batch_size = 128
        self.eval_batch_size = 128

        self.results_dir = "results"
        self.json_summary = None
        self.update(kwargs)
        # default locations of data files
        
        self.pretrain_tfrecords = os.path.join(
            "data", "pretrain_tfrecords/pretrain_data.tfrecord*")
        self.vocab_file = os.path.join("vocab", "vocab.txt")
        self.model_dir = os.path.join(self.results_dir, "models", model_name)
        self.checkpoints_dir = os.path.join(self.model_dir, "checkpoints")
        self.weights_dir = os.path.join(self.model_dir, "weights")
        self.results_txt = os.path.join(self.results_dir, "unsup_results.txt")
        self.results_pkl = os.path.join(self.results_dir, "unsup_results.pkl")
        self.log_dir = os.path.join(self.model_dir, "logs")

        self.max_predictions_per_seq = int((self.mask_prob + 0.005) *
                                           self.max_seq_length)

        # defaults for different-sized model
        if self.model_size == "base":
            self.embedding_size = 768
            self.hidden_size = 768
            self.num_hidden_layers = 12
            self.generator_hidden_size = 0.3333333
            if self.hidden_size % 64 != 0:
                raise ValueError("Hidden size {} should be divisible by 64. Number of attention heads is hidden size {} / 64 ".format(self.hidden_size, self.hidden_size))	
            self.num_attention_heads = int(self.hidden_size / 64.)
        elif self.model_size == "large":
            self.embedding_size = 1024
            self.hidden_size = 1024
            self.num_hidden_layers = 24
            if self.hidden_size % 64 != 0:
                raise ValueError("Hidden size {} should be divisible by 64. Number of attention heads is hidden size {} / 64 ".format(self.hidden_size, self.hidden_size))
            self.num_attention_heads = int(self.hidden_size / 64.)
        elif self.model_size == "small":
            self.embedding_size = 128
            self.hidden_size = 256 # generatr_hidden_size 1.0
            self.num_hidden_layers = 12
            self.generator_hidden_size = 1.0
            if self.hidden_size % 64 != 0:
                raise ValueError("Hidden size {} should be divisible by 64. Number of attention heads is hidden size {} / 64 ".format(self.hidden_size, self.hidden_size))	
            self.num_attention_heads = int(self.hidden_size / 64.)
        elif self.model_size == "small++":
            self.embedding_size = 256
            self.hidden_size = 512 # generator_hidden_size 0.5
            self.num_hidden_layers = 12
            self.generator_hidden_size = 0.5
            if self.hidden_size % 64 != 0:
                raise ValueError("Hidden size {} should be divisible by 64. Number of attention heads is hidden size {} / 64 ".format(self.hidden_size, self.hidden_size))	
            self.num_attention_heads = int(self.hidden_size / 64.)
        else:
            raise ValueError("--model_size : 'small','small++', 'base' and 'large supported only.")
        self.act_func = "gelu"
        self.hidden_dropout_prob = 0.1 
        self.attention_probs_dropout_prob = 0.1

        self.update(kwargs)

    def update(self, kwargs):
        for k, v in kwargs.items():
            if v is not None:
                self.__dict__[k] = v
