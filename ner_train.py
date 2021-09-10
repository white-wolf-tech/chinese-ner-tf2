#coding=utf-8
import tensorflow as tf
import os
from model.ner_model import ner_model
from model.data import gen_data,load_vocs,gen_voc,save_dev,parse_ner_content
import model.data_span as span_data
from model.f1 import compare_f1
from optimization import create_optimizer
from ner_config import nerConfig
from tqdm import tqdm
import argparse
import random

'''
cpu性能参数设置
'''
def set_cpu_performance():
    config = tf.compat.v1.ConfigProto()
    config.intra_op_parallelism_threads = 0
    config.inter_op_parallelism_threads = 0
    os.environ['KMP_BLOCKTIME'] = "1"
    os.environ['KMP_SETTINGS'] = "1"
    os.environ['KMP_AFFINITY'] = "granularity=fine,verbose,compat,1,0"
    os.environ['OMP_NUM_THREADS'] = "8"
    tf.compat.v1.Session(config=config)

train_data_ts = "./train_data/ts/train.json"
dev_data_ts = "./train_data/ts/dev.json"
train_data_cand = "./train_data/ts/train.json"
dev_data_cand = "./train_data/ts/dev.json"

data_voc = "./voc_dir/data.vocab"
label_voc = "./voc_dir/label.vocab"
label_span_voc = "./voc_dir/label_span.vocab"
checkpoint_dir = "checkpoint/"

def build_train_op(config):
    ner = ner_model(config, training=True)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    adv_loss = tf.keras.metrics.Mean(name='adv_loss')
    if config.dynamics_lr:
        optimizer,lr_schedule = create_optimizer(config.lr,
                                    config.num_train_steps,
                                    config.num_warmup_steps,
                                    min_lr_ratio=config.min_lr_rate,
                                    weight_decay_rate=config.decay_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(config.lr)
    #设置检查点
    ckpt = tf.train.Checkpoint(ner=ner, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir + 'trains', max_to_keep=2)
    #恢复旧模型
    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("creat new model....")
    return ner, train_loss, adv_loss, ckpt, ckpt_manager, optimizer

# 载入配置项
config = nerConfig()
# 创建ner训练实例
ner, train_loss, adv_loss, ckpt, ckpt_manager, optimizer = build_train_op(config)
train_loss(0.0)
adv_loss(0.0)


#加入对抗训练
train_span_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                        tf.TensorSpec(shape=(None, None, 11), dtype=tf.int32),
                        tf.TensorSpec(shape=(None, None, 11), dtype=tf.int32)]
@tf.function(input_signature=train_span_signature)
def fgm_train(input_ids, head_labels, tail_labels):
    with tf.GradientTape() as tape0:
        loss = ner(input_ids, head_labels, tail_labels)
    embeddings = [(index, param) for index,param in enumerate(ner.trainable_variables) if "embedding" in param.name]
    # 计算embeding梯度, 只有一层embeding，取第一个值
    embed_index, embedding = embeddings[0][0], embeddings[0][1]
    embed_grads = tape0.gradient(loss, [embedding])
    #转换梯度值为tensor
    embed_grad = tf.zeros_like(embedding) + embed_grads[0]
    #计算扰动因子
    delta = config.epsilon * embed_grad / (tf.math.sqrt(tf.math.reduce_sum(embed_grad**2)) + 1e-8)
    #加入扰动
    ner.trainable_variables[embed_index].assign_add(delta)
    #计算新loss
    with tf.GradientTape() as tape1:
        loss_adv = ner(input_ids, head_labels, tail_labels)
    gradients = tape1.gradient(loss_adv, ner.trainable_variables)
    #恢复embeding层
    ner.trainable_variables[embed_index].assign_sub(delta)
    #梯度更新
    optimizer.apply_gradients(zip(gradients, ner.trainable_variables))
    train_loss(loss)
    adv_loss(loss_adv)

# 解码器采用span方式
train_span_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                        tf.TensorSpec(shape=(None, None, 11), dtype=tf.int32),
                        tf.TensorSpec(shape=(None, None, 11), dtype=tf.int32)]
@tf.function(input_signature=train_span_signature)
def train_span_step(input_ids, head_labels, tail_labels):
    with tf.GradientTape() as tape:
        loss = ner(input_ids, head_labels, tail_labels)
    gradients = tape.gradient(loss, ner.trainable_variables)
    optimizer.apply_gradients(zip(gradients, ner.trainable_variables))
    train_loss(loss)

# 解码器采用crf方式
train_step_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                        tf.TensorSpec(shape=(None, None), dtype=tf.int32)]
@tf.function(input_signature=train_step_signature)
def train_crf_step(input_ids, input_labels):
    with tf.GradientTape() as tape:
        loss = ner(input_ids, input_labels)
    gradients = tape.gradient(loss, ner.trainable_variables)
    optimizer.apply_gradients(zip(gradients, ner.trainable_variables))
    train_loss(loss)


def train(train_set, tokenize):
    train_loss.reset_states()
    adv_loss.reset_states()

    random.shuffle(train_set)
    tq = tqdm(enumerate(train_set))
    for index,batch in tq:
        if tokenize is not None:
            batch_data, batch_label_heads, batch_label_tails = span_data.gen_batch_data_for_train(batch,tokenize)
            fgm_train(batch_data,batch_label_heads, batch_label_tails)
        else:
            batch_data,batch_label = batch
            train_crf_step(batch_data,batch_label)

        tq.set_description('Epoch {} loss {:.16f}, adv_loss {:.16f}'.format(
                                                epoch,
                                                train_loss.result(),
                                                adv_loss.result()))
        if index % 50 == 0 and index > 0:
            save_path = ckpt_manager.save()
            print("Saved checkpoint {}".format(save_path))

def infer(dev_set, data_vocab, labels_vocab, last_f1, tokenize):
    ner = ner_model(config, training=False)
    #从训练的检查点恢复权重
    ckpt = tf.train.Checkpoint(ner=ner)
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir + 'trains')
    #添加expect_partial()关闭优化器相关节点warnning打印
    status = ckpt.restore(latest_ckpt).expect_partial()
    #定义infer函数
    infer_step_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int32)]
    @tf.function(input_signature=infer_step_signature)
    def predict(input_ids):
        L = ner(input_ids)
        return L

    infers = []
    datas = []
    infer_json = []
    for batch_val in dev_set:
        if tokenize is not None:
            batch_val_data = span_data.gen_batch_data_for_train(batch_val, tokenize, False)
        else:
            batch_val_data,_ = batch_val
        infers.append(predict(batch_val_data))
        datas.append(batch_val_data)
    if config.decode_type == 'crf':
        infer_json = parse_ner_content(infers,datas,[data_vocab,labels_vocab],config.label_train_type)
    else:
        for index, item in enumerate(dev_set):
            infer_batch = infers[index]
            batch_result = span_data.parse_span_ner(item, infer_batch, tokenize)
            infer_json.extend(batch_result)
    with open("test/infer.json", 'w') as wf:
        wf.write('\n'.join(infer_json))
    _,average_f1 = compare_f1('test/infer.json', 'test/dev.json')
    if last_f1 < average_f1:
        last_f1 = average_f1
        print("f1 improve save model...")
        tf.saved_model.save(ner, checkpoint_dir + 'infers/')
        with open(checkpoint_dir + 'infers/best_f1', 'w') as wf:
            wf.write(str(last_f1))
    print('average f1 is {}'.format(average_f1))
    return last_f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parse')
    parser.add_argument('--training',type=str,default="train")
    args = parser.parse_args()
    if not config.use_gpu:
        set_cpu_performance()
    #选择数据集
    train_file = train_data_cand
    dev_file = dev_data_cand
    if config.train_type == "ts":
        train_file = train_data_ts
        dev_file = dev_data_ts
    #检查字典是否存在或者重新生成字典标志位设置
    if not os.path.isfile(data_voc) or config.reset_vocab:
        gen_voc([train_file, dev_file],save_data_file=data_voc,save_label_file=label_voc)
        save_dev(dev_file,'test/dev.json')
    #载入字典文件
    data_vocab, labels_vocab, tokenize = None, None, None
    if config.decode_type == 'crf':
        data_vocab,labels_vocab = load_vocs(data_voc,label_voc)
        #载入训练数据和测试数据
        train_set = gen_data(dataset=train_file,
            vocabs=[data_vocab,labels_vocab],
            batch_size=config.batch_size,
            is_training=True,
            label_train_type=config.label_train_type)
        dev_set = gen_data(dataset = dev_file,
            vocabs=[data_vocab,labels_vocab],
            batch_size=config.batch_size,
            is_training=False,
            label_train_type=config.label_train_type)
    elif config.decode_type == 'span':
        tokenize = span_data.Tokenize(data_voc, label_span_voc)
        train_set = span_data.load_json_data(train_file, config.batch_size)
        dev_set = span_data.load_json_data(dev_file, 1) #infer时,设置batch设为1
    else:
        raise ValueError
    #训练和测试
    training = (args.training == "train")
    last_f1 = 0.0
    if training:
        #开始训练
        for epoch in range(config.epoch):
            train(train_set, tokenize)
            last_f1 = infer(dev_set, data_vocab, labels_vocab, last_f1, tokenize)
    else:
        infer(dev_set, data_vocab, labels_vocab, last_f1, tokenize)
