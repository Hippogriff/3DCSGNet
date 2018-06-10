"""
Testing fix len programs
"""
import numpy as np
import torch
import os
import json
from torch.autograd.variable import Variable
from src.Utils import read_config
from src.Generator.generator import Generator
from src.Models.models import CsgNet, ParseModelOutput
from src.Utils.train_utils import prepare_input_op
import time
import sys

if len(sys.argv) > 1:
    config = read_config.Config(sys.argv[1])
else:
    config = read_config.Config("config.yml")
print(config.config)


data_labels_paths = {3: "data/one_op/expressions.txt",
                     5: "data/two_ops/expressions.txt",
                     7: "data/three_ops/expressions.txt"}
dataset_sizes = {3: [110000, 20000],
                 5: [220000, 40000],
                 7: [440000, 80000]}

test_gen_objs = {}
types_prog = len(dataset_sizes.keys())
max_len = max(data_labels_paths.keys())

generator = Generator(data_labels_paths=data_labels_paths,
                      batch_size=config.batch_size,
                      time_steps=max(data_labels_paths.keys()),
                      stack_size=max(data_labels_paths.keys()) // 2 + 1)


imitate_net = CsgNet(grid_shape=[64, 64, 64], dropout=config.dropout,
                     mode=config.mode, timesteps=max_len,
                     num_draws=len(generator.unique_draw),
                     in_sz=config.input_size,
                     hd_sz=config.hidden_size,
                     stack_len=config.top_k)

cuda_devices = torch.cuda.device_count()
if torch.cuda.device_count() > 1:
    imitate_net.cuda_devices = torch.cuda.device_count()
    print("using multi gpus", flush=True)
    imitate_net = torch.nn.DataParallel(imitate_net, dim=0)
    imitate_net.load_state_dict(torch.load(config.pretrain_modelpath))
else:
    weights = torch.load(config.pretrain_modelpath)
    new_weights = {}
    for k in weights.keys():
        if k.startswith("module"):
            new_weights[k[7:]] = weights[k]
    imitate_net.load_state_dict(new_weights)


imitate_net.cuda()

for param in imitate_net.parameters():
    param.requires_grad = True

config.test_size = sum((dataset_sizes[k][1] // config.batch_size) * config.batch_size
                       for k in dataset_sizes.keys())

for k in data_labels_paths.keys():
    # if using multi gpu training, train and test batch size should be multiple of
    # number of GPU edvices.
    test_batch_size = config.batch_size
    test_gen_objs[k] = generator.get_test_data(test_batch_size,
                                               k,
                                               num_train_images=dataset_sizes[k][0],
                                               num_test_images=dataset_sizes[k][1],
                                               if_primitives=True,
                                               if_jitter=False)

Target_expressions = []
Predicted_expressions = []
parser = ParseModelOutput(generator.unique_draw, max_len // 2 + 1, max_len, [64, 64, 64], primitives=generator.primitives)
imitate_net.eval()
Rs = 0
t1 = time.time()
IOU = {}
total_iou = 0
for k in data_labels_paths.keys():
    Rs = 0.0
    for batch_idx in range(dataset_sizes[k][1] // config.batch_size):
        data_, labels = next(test_gen_objs[k])
        data_ = data_[:, :, 0:config.top_k + 1, :, :, :]
        one_hot_labels = prepare_input_op(labels, len(generator.unique_draw))
        one_hot_labels = Variable(torch.from_numpy(one_hot_labels), volatile=True).cuda()
        data = Variable(torch.from_numpy(data_)).cuda()
        labels = Variable(torch.from_numpy(labels)).cuda()

        # This is for data parallelism purpose
        data = data.permute(1, 0, 2, 3, 4, 5)

        if cuda_devices > 1:
            outputs = imitate_net.module.test([data, one_hot_labels, max_len])
        else:
            outputs = imitate_net.test([data, one_hot_labels, max_len])

        stack, _, expressions = parser.get_final_canvas(outputs, if_pred_images=True,
                                                  if_just_expressions=False)
        Predicted_expressions += expressions
        target_expressions = parser.labels2exps(labels, k)
        Target_expressions += target_expressions
        # stacks = parser.expression2stack(expressions)
        data_ = data_[-1, :, 0, :, :, :]
        R = np.sum(np.logical_and(stack, data_), (1, 2, 3)) / (np.sum(
            np.logical_or(stack, data_), (1, 2, 3)) + 1)
        Rs += np.sum(R)
    total_iou += Rs
    IOU[k] = Rs / ((dataset_sizes[k][1] // config.batch_size) * config.batch_size)
    print("IOU for {} len program: ".format(k), IOU[k])


total_iou = total_iou / config.test_size
print ("total IOU score: ", total_iou)
results = {"total_iou": total_iou, "iou": IOU}

results_path = "trained_models/results/{}/".format(config.pretrain_modelpath)
os.makedirs(os.path.dirname(results_path), exist_ok=True)

with open(results_path + "pred.txt", "w") as file:
    for p in Predicted_expressions:
        file.write(p + "\n")

with open(results_path + "target.txt", "w") as file:
    for p in Target_expressions:
        file.write(p + "\n")

with open(results_path + "results.org", 'w') as outfile:
    json.dump(results, outfile)
