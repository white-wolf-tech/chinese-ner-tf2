# coding=utf-8
class nerConfig():
    def __init__(
        self,
        vocab_size=3747,
        tgt_size=44,
        max_length=256,
        n_layer=4,
        head=8,
        embed=256,
        ffw_rate=4,
        a_dropout=0.2,
        layer_norm_epsilon=1e-5,
        batch_size=16,
        lr=1e-4,
        dynamics_lr=True,
        rnn=True,
        rnn_unit=384,
        rnn_dropout=0.2,
        epoch=1000,
        label_train_type='BIES',
        train_type='ts',
        reset_vocab=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.tgt_size = tgt_size
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
        self.dynamics_lr = dynamics_lr
        self.rnn = rnn
        self.rnn_unit = rnn_unit
        self.rnn_dropout = rnn_dropout
        self.label_train_type = label_train_type
        self.train_type = train_type
        self.reset_vocab = reset_vocab