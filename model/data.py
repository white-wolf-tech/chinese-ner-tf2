#coding=utf-8
import random
import math
import numpy as np
import json
import tensorflow as tf
'''
字典生成与载入相关代码
'''
def load_json_data(filename):
    with open(filename,'rb') as file_r:
        data_lists = []
        label_lists = []
        data_list = file_r.read().strip().decode('utf-8').split('\n')
        for items in data_list:
            items = json.loads(items)
            seq = items["text"]
            label = items["label"]
            data_lists.append(seq)
            label_lists.append(label)
        return data_lists,label_lists

def save_dev(dev_file,save_path):
    _, labels = load_json_data(dev_file)
    save_list = []
    for data in labels:
        data = json.dumps({'label':data},ensure_ascii=False,sort_keys=False)
        save_list.append(data)
    with open(save_path,'w') as wf:
        wf.write("\n".join(save_list))

def gen_voc(data_vocs,save_data_file=None,save_label_file=None):
    src_s = []
    label_s = []
    for data in data_vocs:
        data_lists,label_lists = load_json_data(data)
        src_s = src_s + data_lists
        label_s = label_s + label_lists
    src_list = ["PAD","EOS","UNK"]
    label_list = ["PAD","EOS","UNK","O"]
    max_len = 0
    for inx,item in enumerate(src_s):
        datas = item.strip()
        if len(datas) > max_len:
            max_len =len(datas) 
        for item_data in datas:
            if item_data == '':
                continue 
            if item_data not in src_list:
                src_list.append(item_data)
    label_temp = []
    for label in label_s:
        for item_label in label.keys():
            if item_label == '':
                continue
            if item_label not in label_temp:
                label_temp.append(item_label)
    for _label in label_temp:
        current_list = ['B_' + _label,'I_'+_label,'E_'+_label,'S_'+_label]
        label_list.extend(current_list)
    with open(save_data_file,'w') as wf:
        wf.write('\n'.join(src_list))
    with open(save_label_file,'w') as wf:
        wf.write('\n'.join(label_list))
    print("max data len is {}".format(max_len))

def load_vocs(save_data_file,save_label_file):
    with open(save_data_file,'r') as sfr:
        datas_vocab = sfr.read().strip().split('\n')
    with open(save_label_file,'r') as tfr:
        labels_vocab = tfr.read().strip().split('\n')
    datas_vocab = dict(zip(datas_vocab,list(range(len(datas_vocab)))))
    labels_vocab = dict(zip(labels_vocab,list(range(len(labels_vocab)))))
    return datas_vocab,labels_vocab

'''
模型训练数据处理相关代码
'''
def pading_max_len(batch_data,batch_label,max_len):
    PAD = 0
    for inx,item in enumerate(batch_data):
        data_len = len(item)
        if data_len < max_len:
            batch_data[inx] = item + [PAD] * (max_len - data_len)
            batch_label[inx] = batch_label[inx] + [PAD] * (max_len - data_len)
    return [tf.convert_to_tensor(batch_data),tf.convert_to_tensor(batch_label)]

def get_label(label,label_voc,seq_len,label_train_type):
    current_types = label.keys()
    label_init = [label_voc["O"]] * seq_len
    if len(current_types) == 0:
        return label_init
    for label_type in current_types:
        id_data = label_voc["B_" + label_type]
        entities = label[label_type]
        for item in entities.keys():
            entity = entities[item]
            for b_e in entity:
                start_pos = b_e[0]
                end_pos = b_e[1]
                replace_list = []
                if label_train_type == 'BIES':
                    if end_pos - start_pos == 0:
                        replace_list = [id_data + 3]
                    elif end_pos - start_pos == 1:
                        replace_list = [id_data, id_data + 2]
                    elif end_pos - start_pos > 1:
                        inner = [id_data + 1] * (end_pos - start_pos - 1) #忽略末尾
                        replace_list = [id_data] + inner + [id_data + 2]
                elif label_train_type == 'BI':
                    inner = [id_data + 1] * (end_pos - start_pos)
                    replace_list = [id_data] + inner
                for inx,it in enumerate(replace_list):
                    label_init[start_pos+inx] = replace_list[inx]
    return label_init

def gen_data(dataset=None,
            vocabs=None,
            batch_size=128,
            is_training=True,
            label_train_type='BIES'):
    datas,labels = load_json_data(dataset)
    batch_data = []
    batch_label = []
    max_len = 0
    i = 0
    datas_res = []
    for index,item in enumerate(datas):
        data = list(item.strip())
        label_ids = get_label(labels[index],vocabs[-1],len(data),label_train_type)
        data_ids = []
        for data_item in data:
            if data_item in vocabs[0].keys():
                data_ids.append(vocabs[0][data_item])
            else:
                data_ids.append(vocabs[0]["UNK"])
        data_ids.append(vocabs[0]["EOS"])
        data_ids.append(vocabs[0]["PAD"])
        label_ids.append(vocabs[-1]["EOS"])
        label_ids.append(vocabs[-1]["PAD"])
        current_len = len(data_ids)
        if current_len > max_len:
            max_len = current_len
        batch_data.append(data_ids)
        batch_label.append(label_ids)
        if len(batch_data) == batch_size or index == len(datas) - 1:
            datas_res.append(pading_max_len(batch_data,batch_label,max_len))
            batch_data = []
            batch_label = []
    return datas_res

'''
模型推理相关解码代码
'''
def init_resdict(voc):
    res_dict = {}
    for key in voc.keys():
        if 'UNK' not in key and "PAD" not in key \
        and "EOS" not in key and "O" not in key \
        and 'I_' not in key and 'E_' not in key \
        and 'S_' not in key:
            key = key.split("B_")[-1]
            res_dict[key] = {}
    return res_dict

def get_pos_type(test_entity,id2label,label_mark_type):
    ner_type = 'O'
    try:
        if len(test_entity) == 1:
            if label_mark_type == 'BIES':
                dict_pos = test_entity[0][-1] - 3
                ner_type = id2label[dict_pos].split("_")[-1]
            elif label_mark_type == 'BI':
                dict_pos = test_entity[0][-1]
                ner_type = id2label[dict_pos].split("_")[-1]
        else:
            if label_mark_type == 'BIES':
                dict_pos = test_entity[-1][-1] - 2
                ner_type = id2label[dict_pos].split("_")[-1]
            elif label_mark_type == 'BI':
                dict_pos = test_entity[-1][-1] - 1
                ner_type = id2label[dict_pos].split("_")[-1]
    except Exception as e:
        pass
    return ner_type

def extract_entity(entities,res_dict,id2data,id2label,label_mark_type):
    PAD = 0
    EOS = 1
    UNK = 2
    for indx,entity in enumerate(entities):
        ner_type = get_pos_type(entity,id2label,label_mark_type)
        if ner_type not in res_dict.keys():
            continue
        entity_tmp = []
        for item in entity:
            entity_id = item[1]
            if entity_id in [EOS,PAD,UNK]:
                continue
            else:
                entity_tmp.append(id2data[entity_id])
        entity_content = ''.join(entity_tmp)
        entity_indx = [item[0] for item in entity]
        if len(entity_indx) == 1:
            entity_indx = [entity_indx[0],entity_indx[0]]
        elif len(entity_indx) == 2:
            pass
        elif len(entity_indx) > 2:
            entity_indx = [entity_indx[0],entity_indx[-1]]
        if entity_content not in res_dict[ner_type].keys():
            res_dict[ner_type][entity_content] = [entity_indx]
        else:
            if entity_indx not in res_dict[ner_type][entity_content]:
                res_dict[ner_type][entity_content].append(entity_indx)
    new_res = {}
    for key in res_dict.keys():
        if len(res_dict[key]) == 0:
            pass
        else:
            new_res[key] = res_dict[key] 
    return new_res

def parse_ner_content(infer_batchs,batch_data,vocs,label_mark_type):
    label_voc = vocs[-1]
    data_voc = vocs[0]
    id2data = dict(zip(data_voc.values(),data_voc.keys()))
    id2label = dict(zip(label_voc.values(),label_voc.keys()))
    infers = []
    for index,item in enumerate(infer_batchs): #batch level
        infer_labels = item
        test_datas = batch_data[index]
        for inx,infer_item in enumerate(infer_labels): #sentence level
            infer_item = infer_item.numpy().tolist()
            test_data_item = test_datas[inx].numpy().tolist()

            infer_entities = []
            in_entity = []
            '''
            解析句子中的实体位置和类型
            '''
            for indx,infer_id in enumerate(infer_item):
                test_data_it = test_data_item[indx]
                current_label = id2label[infer_id]
                in_entity.append((indx,test_data_it,infer_id))
                if label_mark_type == 'BI':
                    if current_label == 'O':
                        if indx >= 1:
                            last_infer = id2label[infer_item[indx-1]]
                            if 'I_' in last_infer or 'B_' in last_infer:
                                del in_entity[-1]
                                infer_entities.append(in_entity)
                                in_entity = []
                            else:
                                in_entity = []
                        else:
                            in_entity = []
                    elif 'B_' in current_label:
                        if indx >= 1:
                            last_infer = id2label[infer_item[indx-1]]
                            if 'I_' in last_infer:
                                del in_entity[-1]
                                infer_entities.append(in_entity)
                                in_entity = []
                    elif current_label == 'EOS':
                        del in_entity[-1]
                        if len(in_entity) > 0:
                            infer_entities.append(in_entity)
                            in_entity = []
                            break
                elif label_mark_type == 'BIES':
                    if current_label == 'O':
                        if indx >= 1:
                            last_infer = id2label[infer_item[indx-1]]
                            if 'E_' in last_infer or 'S_' in last_infer:
                                del in_entity[-1]
                                infer_entities.append(in_entity)
                                in_entity = []
                            else:
                                in_entity = []
                        else:
                            in_entity = []
                    elif 'B_' in current_label:
                        if indx >= 1:
                            last_infer = id2label[infer_item[indx-1]]
                            if 'S_' in last_infer or 'E_' in last_infer:
                                del in_entity[-1]
                                infer_entities.append(in_entity)
                                in_entity = []
                    elif 'S_' in current_label:
                        infer_entities.append(in_entity[-1])
                        in_entity = []
                    elif current_label == 'EOS':
                        del in_entity[-1]
                        if len(in_entity) > 0:
                            infer_entities.append(in_entity)
                            in_entity = []
                            break
            '''
            计算所有实体数量
            '''
            infer_result = init_resdict(label_voc)
            infer_result = extract_entity(infer_entities, infer_result, id2data, id2label,label_mark_type)
            infers.append(json.dumps({'label' :infer_result},ensure_ascii=False,sort_keys=False))
    return infers
