import tensorflow as tf

model_infer_saved_path = 'checkpoint/infers/'
loaded = tf.saved_model.load(model_infer_saved_path)
#寻找输出函数
print(list(loaded.signatures.keys())) #serving_default
#获得输出函数
infer_model = loaded.signatures["serving_default"]
#获取输出节点
print(infer_model.structured_outputs)
output_name = list(infer_model.structured_outputs.keys())[0]
#模型推理代码
test_data = tf.constant([[4,5,6,7,8,9,1,0]])
labeling = infer_model(test_data)[output_name]
print("predict:{}".format(labeling))