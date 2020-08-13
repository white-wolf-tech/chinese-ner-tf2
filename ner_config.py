# coding=utf-8
class nerConfig():
    def __init__(
        self,
        vocab_size=3747,
        tgt_size=44,
        span_tgt = 11,
        max_length=2048,
        n_layer=2,
        head=6,
        embed=384,
        ffw_rate=4,
        a_dropout=0.1,
        layer_norm_epsilon=1e-5,
        batch_size=8,
        lr=5e-5,
        num_train_steps=160000,
        num_warmup_steps=80000,
        dynamics_lr=True,
        rnn=False,
        rnn_unit=384,
        ff_dropout=0.2,
        epoch=1000,
        label_train_type='BIES',
        train_type='ts',
        enable_reltf=True,
        decode_type='span',
        reset_vocab=False,
        use_gpu=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.tgt_size = tgt_size
        self.span_tgt = span_tgt
        self.max_length = max_length
        self.embed = embed
        self.n_layer = n_layer
        self.head = head
        self.ffw_rate= ffw_rate
        self.a_dropout = a_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr
        self.num_warmup_steps = num_warmup_steps
        self.num_train_steps = num_train_steps
        self.rnn = rnn #是否使用rnn层
        self.rnn_unit = rnn_unit #rnn单元数目
        self.ff_dropout = ff_dropout #ffw层dropout
        self.label_train_type = label_train_type #使用CRF时，标签格式
        self.train_type = train_type #选择待训练的数据集
        self.reset_vocab = reset_vocab #是否重新生成字典
        self.use_gpu = use_gpu #是否使用gpu
        self.enable_reltf = enable_reltf #是否使用relTransformer架构
        self.decode_type = decode_type #选择span还是crf