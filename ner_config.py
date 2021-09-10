# coding=utf-8
class nerConfig():
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.vocab_size = 3747 #字典大小
        self.tgt_size = 44 #使用crf方式标签字典大小
        self.span_tgt = 11 #使用span方式标签字典大小
        self.max_length = 2048 #ner提取文本最大长度
        self.embed = 512 #transformer embedding大小
        self.n_layer = 4 #transformer层数
        self.head = 8   #自注意力头大小
        self.a_dropout = 0.1 #自注意力内部dropout大小
        self.layer_norm_epsilon = 1e-5 #ln层参数
        self.ffw_rate= 4 #ffw两层之间倍率
        self.lr = 5e-4 #学习率设置
        self.dynamics_lr = True #是否使用warmup方式
        self.scale = False #是否给attention score加scale
        self.min_lr_rate = 0.005 #截止学习率是初始学习率的多少
        self.decay_rate = 0.99 #学习率衰减率
        self.num_warmup_steps = 11000
        self.num_train_steps = 22000
        self.rnn = False #是否使用rnn层
        self.rnn_unit = 384 #rnn单元数目
        self.ff_dropout = 0.2 #ffw层dropout
        self.epoch = 1000
        self.batch_size = 8
        self.train_type = 'ts' #选择待训练的数据集
        self.reset_vocab = False #是否重新生成字典
        self.use_gpu = False #是否使用gpu
        self.enable_reltf = True #是否使用relTransformer架构
        self.decode_type = 'span' #选择span还是crf
        self.label_train_type = 'BIES' #使用CRF时，标签格式
        self.epsilon = 0.5 #对抗训练，计算扰动因子超参
