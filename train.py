"""
Train the network using mixture of programs.
"""
import sys
import numpy as np
import torch
import torch.optim as optim
from tensorboard_logger import configure, log_value
from torch.autograd.variable import Variable
from src.Utils import read_config
from src.Generator.generator import Generator
from src.Models.loss import losses_joint
from src.Models.models import CsgNet, ParseModelOutput
from src.Utils.learn_utils import LearningRate
from src.Utils.train_utils import prepare_input_op, Callbacks

if len(sys.argv) > 1:
    config = read_config.Config(sys.argv[1])
else:
    config = read_config.Config("config.yml")

model_name = config.model_path.format(config.proportion,
                                      config.top_k,
                                      config.hidden_size,
                                      config.batch_size,
                                      config.optim, config.lr,
                                      config.weight_decay,
                                      config.dropout,
                                      "mix",
                                      config.mode)
print(config.config)

config.write_config("log/configs/{}_config.json".format(model_name))
configure("log/tensorboard/{}".format(model_name), flush_secs=5)


callback = Callbacks(config.batch_size, "log/db/{}".format(model_name))
callback.add_element(["train_loss", "test_loss", "train_mse", "test_mse"])

data_labels_paths = {3: "data/one_op/expressions.txt",
                     5: "data/two_ops/expressions.txt",
                     7: "data/three_ops/expressions.txt"}

proportion = config.proportion  # proportion is in percentage. vary from [1, 100].

# First is training size and second is validation size per program length
dataset_sizes = {3: [proportion * 1000, proportion * 250],
                 5: [proportion * 2000, proportion * 500],
                 7: [proportion * 4000, proportion * 100]}

config.train_size = sum(dataset_sizes[k][0] for k in dataset_sizes.keys())
config.test_size = sum(dataset_sizes[k][1] for k in dataset_sizes.keys())
types_prog = len(dataset_sizes)

generator = Generator(data_labels_paths=data_labels_paths,
                      batch_size=config.batch_size,
                      time_steps=max(data_labels_paths.keys()),
                      stack_size=max(data_labels_paths.keys()) // 2 + 1)

imitate_net = CsgNet(grid_shape=[64, 64, 64], dropout=config.dropout,
                     mode=config.mode, timesteps=max(data_labels_paths.keys()),
                     num_draws=len(generator.unique_draw),
                     in_sz=config.input_size,
                     hd_sz=config.hidden_size,
                     stack_len=config.top_k)

# If you want to use multiple GPUs for training.
cuda_devices = torch.cuda.device_count()
if torch.cuda.device_count() > 1:
    imitate_net.cuda_devices = torch.cuda.device_count()
    print("using multi gpus", flush=True)
    imitate_net = torch.nn.DataParallel(imitate_net, device_ids=[0, 1], dim=0)
imitate_net.cuda()

if config.preload_model:
    imitate_net.load_state_dict(torch.load(config.pretrain_modelpath))

for param in imitate_net.parameters():
    param.requires_grad = True

if config.optim == "sgd":
    optimizer = optim.SGD(
        [para for para in imitate_net.parameters() if para.requires_grad],
        weight_decay=config.weight_decay,
        momentum=0.9, lr=config.lr, nesterov=False)

elif config.optim == "adam":
    optimizer = optim.Adam(
        [para for para in imitate_net.parameters() if para.requires_grad],
        weight_decay=config.weight_decay, lr=config.lr)

reduce_plat = LearningRate(optimizer, init_lr=config.lr, lr_dacay_fact=0.2,
                           lr_decay_epoch=3, patience=config.patience)

train_gen_objs = {}
test_gen_objs = {}

# Prefetching minibatches
for k in data_labels_paths.keys():
    # if using multi gpu training, train and test batch size should be multiple of
    # number of GPU edvices.
    train_batch_size = config.batch_size // types_prog
    test_batch_size = config.batch_size // types_prog
    train_gen_objs[k] = generator.get_train_data(train_batch_size,
                                                 k,
                                                 num_train_images=dataset_sizes[k][0],
                                                 if_primitives=True,
                                                 if_jitter=False)
    test_gen_objs[k] = generator.get_test_data(test_batch_size,
                                               k,
                                               num_train_images=dataset_sizes[k][0],
                                               num_test_images=dataset_sizes[k][1],
                                               if_primitives=True,
                                               if_jitter=False)

prev_test_loss = 1e20
prev_test_reward = 0
test_size = config.test_size
batch_size = config.batch_size
for epoch in range(0, config.epochs):
    train_loss = 0
    Accuracies = []
    imitate_net.train()
    # Number of times to accumulate gradients
    num_accums = config.num_traj
    for batch_idx in range(config.train_size // (config.batch_size * config.num_traj)):
        optimizer.zero_grad()
        loss_sum = Variable(torch.zeros(1)).cuda().data
        for _ in range(num_accums):
            for k in data_labels_paths.keys():
                data, labels = next(train_gen_objs[k])

                data = data[:, :, 0:config.top_k + 1, :, :, :]
                one_hot_labels = prepare_input_op(labels, len(generator.unique_draw))
                one_hot_labels = Variable(torch.from_numpy(one_hot_labels)).cuda()
                data = Variable(torch.from_numpy(data)).cuda()
                labels = Variable(torch.from_numpy(labels)).cuda()
                data = data.permute(1, 0, 2, 3, 4, 5)

                # forward pass
                outputs = imitate_net([data, one_hot_labels, k])

                loss = losses_joint(outputs, labels, time_steps=k + 1) / types_prog / \
                       num_accums
                loss.backward()
                loss_sum += loss.data

        # Clip the gradient to fixed value to stabilize training.
        torch.nn.utils.clip_grad_norm(imitate_net.parameters(), 20)
        optimizer.step()
        l = loss_sum
        train_loss += l
        log_value('train_loss_batch', l.cpu().numpy(), epoch * (
            config.train_size //
            (config.batch_size * num_accums)) + batch_idx)
    mean_train_loss = train_loss / (config.train_size // (config.batch_size * num_accums))
    log_value('train_loss', mean_train_loss.cpu().numpy(), epoch)
    del data, loss, loss_sum, train_loss, outputs

    test_losses = 0
    imitate_net.eval()
    test_reward = 0
    for batch_idx in range(config.test_size // config.batch_size):
        for k in data_labels_paths.keys():
            parser = ParseModelOutput(generator.unique_draw,
                                      stack_size=(k + 1) // 2 + 1,
                                      steps=k,
                                      canvas_shape=[64, 64, 64],
                                      primitives=generator.primitives)
            data_, labels = next(test_gen_objs[k])

            one_hot_labels = prepare_input_op(labels, len(generator.unique_draw))
            one_hot_labels = Variable(torch.from_numpy(one_hot_labels)).cuda()
            data = Variable(torch.from_numpy(data_[:, :, 0:config.top_k + 1, :, :, :]), volatile=True).cuda()
            data = data.permute(1, 0, 2, 3, 4, 5)
            labels = Variable(torch.from_numpy(labels)).cuda()

            test_output = imitate_net([data, one_hot_labels, k])

            l = losses_joint(test_output, labels, time_steps=k + 1).data
            test_losses += l

            if cuda_devices > 1:
                test_output = imitate_net.module.test([data, one_hot_labels, k])
            else:
                test_output = imitate_net.test([data, one_hot_labels, k])

            stack, _, _ = parser.get_final_canvas(test_output, if_pred_images=True,
                                                  if_just_expressions=False)
            data_ = data_[-1, :, 0, :, :, :]
            R = np.sum(np.logical_and(stack, data_), (1, 2, 3)) / (
            np.sum(np.logical_or(stack, data_), (1, 2, 3)) + 1)
            test_reward += np.sum(R)

    test_reward = test_reward / (test_size // batch_size) / ((batch_size // types_prog) * types_prog)

    test_loss = test_losses.cpu().numpy() / (config.test_size // config.batch_size) / types_prog
    log_value('test_loss', test_loss, epoch)
    log_value('test_IOU', test_reward / (config.test_size // config.batch_size), epoch)
    callback.add_value({
        "test_loss": test_loss,
    })
    print ("Average test IOU: {} at {} epoch".format(test_reward, epoch))
    if config.if_schedule:
        reduce_plat.reduce_on_plateu(-test_reward)

    del test_losses, test_output
    if test_reward > prev_test_reward:
        torch.save(imitate_net.state_dict(),
                   "trained_models/{}.pth".format(model_name))
        prev_test_reward = test_reward
    callback.dump_all()