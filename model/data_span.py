#coding=utf-8
import random
import os
import json
import numpy as np
'''
生成ner标签字典
'''
def gen_ner_vocab(dirs, save_file, save_local_dict, save_dev_file):
    filenames = [os.path.join(dirs,file) for file in os.listdir(dirs) if file.endswith('.json')]
    local_ner_key = []
    label_keys = []
    dev = []
    is_dev = False
    for filename in filenames:
        if 'dev' in filename:
            is_dev = True
        with open(filename,'r') as file_r:
            data_list = file_r.read().strip().split('\n')
            for items in data_list:
                items = json.loads(items)
                text = items["text"]
                label = items["label"]
                if is_dev:
                    dev.append(json.dumps({"label":label},ensure_ascii=False,sort_keys=False))
                label_keys.extend(label.keys())
                local_ner_key.extend(list(text))
            label_keys = list(set(label_keys))
            local_ner_key = list(set(local_ner_key))
    label_keys.insert(0,"PAD")
    local_ner_key.insert(0,'PAD')
    local_ner_key.insert(1,'EOS')
    local_ner_key.insert(2,'UNK')
    with open(save_file,'w') as wf:
        wf.write("\n".join(label_keys))
    with open(save_local_dict,'w') as wf:
        wf.write("\n".join(local_ner_key))
    if is_dev:
        with open(save_dev_file,'w') as wfd:
            wfd.write("\n".join(dev))
'''
载入ner数据
'''
def load_json_data(filename,batch_size):
    with open(filename,'r') as file_r:
        data_res = []
        data_batch = []
        max_len = 0
        data_list = file_r.read().strip().split('\n')
        for items in data_list:
            items = json.loads(items)
            text = items["text"]
            label = items["label"]
            if len(text) > max_len:
                max_len = len(text)
            data_batch.append((text,label))
            if len(data_batch) == batch_size:
                data_res.append((data_batch,max_len))
                data_batch = []
                max_len = 0
        if len(data_batch) > 0:
            data_res.append((data_batch,max_len))
        return data_res

class Tokenize(object):
    """docstring for tokenize"""
    def __init__(self, vocab_data, vocab_label):
        super(Tokenize, self).__init__()
        with open(vocab_data,'r') as sfr:
            datas_vocab = sfr.read().strip().split('\n')
        with open(vocab_label,'r') as tfr:
            labels_vocab = tfr.read().strip().split('\n')
        self.data_voc = dict(zip(datas_vocab,list(range(len(datas_vocab)))))
        self.label_voc = dict(zip(labels_vocab,list(range(len(labels_vocab)))))
        self.id2label = dict(zip(list(range(len(labels_vocab))),labels_vocab))
        self.label_len = len(self.label_voc)
    def encode(self, tokens):
        token_ids = []
        for token in tokens:
            if token in self.data_voc:
                token_ids.append(self.data_voc[token])
            else:
                token_ids.append(self.data_voc["UNK"])
        return token_ids

    def padding(self, token_ids, target_len):
        EOS = self.data_voc["EOS"]
        PAD = self.data_voc["PAD"]
        if len(token_ids) > target_len:
            token_ids = token_ids[:target_len - 1] + [EOS]
        elif len(token_ids) < target_len - 1:
            token_ids = token_ids + [EOS] + [PAD]*(target_len -1 - len(token_ids))
        else:
            token_ids = token_ids + [EOS]
        return token_ids

    def find_label_id(self,label_type):
        if label_type in self.label_voc:
            return self.label_voc[label_type]
        else:
            return self.label_voc["PAD"]

def gen_batch_data_for_train(batch_data,tokenize,is_training=True):
    batch_,max_len = batch_data
    tokens = []
    head_golds = []
    tail_golds = []
    span_golds = []
    for data in batch_:
        text,label = data 
        ids = tokenize.encode(text)
        ids = tokenize.padding(ids,max_len + 1) #尾增加EOS
        tokens.append(ids)
        if is_training:
            heads = np.zeros([max_len + 1, tokenize.label_len])
            tails = np.zeros([max_len + 1, tokenize.label_len])
            for label_key in label:
                ner_type_id = tokenize.find_label_id(label_key)
                for entity in list(label[label_key].values()):
                    for i_entity in entity:
                        heads[i_entity[0]][ner_type_id] = 1
                        tails[i_entity[1]][ner_type_id] = 1
            head_golds.append(heads)
            tail_golds.append(tails)
    if is_training:
        return np.array(tokens), np.array(head_golds), np.array(tail_golds)
    else:
        return np.array(tokens)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def parse_span_ner(batch, output, tokenize, thr=0.5):
    head_logits = output[0] #[batch, seq_len, label_size]
    tail_logits = output[1] #[batch, seq_len, label_size]
    #print(head_logits[0] > 0.5)
    #print(tail_logits[0] > 0.5)
    batch_text,max_len = batch
    result = []
    for index,head_it in enumerate(head_logits):
        text,_ = batch_text[index]
        tail_it = tail_logits[index]
        heads = np.where(head_it > thr)
        tails = np.where(tail_it > thr)
        label = {}
        head_positions = heads[0]
        tail_positions = tails[0]
        head_types = heads[1]
        tail_types = tails[1]
        for head_index,head in enumerate(head_positions):
            head_type = head_types[head_index]
            tail = None
            tail_type = None
            for tail_index,it in  enumerate(tail_positions):
                if it >= head:
                    tail = it
                    tail_type = tail_types[tail_index]
                    break
            if tail == None:
                break
            else:
                entity = text[head : tail + 1]
                if head_type == tail_type:
                    entity_type = tokenize.id2label[tail_type]
                    if entity_type not in label:
                        label[entity_type] = {entity:[[head,tail]]}
                    else:
                        if entity not in label[entity_type]:
                            label[entity_type][entity] = [[head,tail]]
                        else:
                            label[entity_type][entity].append([head,tail])
        result.append(json.dumps({"label":label}, 
                                 ensure_ascii=False,
                                 sort_keys=False,
                                 cls=NpEncoder))
    return result

if __name__ == '__main__':
    gen_ner_vocab('../train_data/ts/', '../voc_dir/label_span.vocab', '../voc_dir/data_spand.vocab', '../test/dev_span.json')