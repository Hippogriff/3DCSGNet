import torch
import torch.nn as nn
from torch.autograd import Variable
from typing import List

nllloss = nn.NLLLoss()


def Losses(out, labels, time_steps, mode=0):
    """
    Defines losses for different modes
    :param out: output from the network
    :param labels: Ground truth labels
    :param time_steps: Length of the program
    :param mode: model type
    :return Loss: Sum of categoricam losses 
    """
    shape_loss = Variable(torch.zeros(1)).cuda()
    size_loss = Variable(torch.zeros(1)).cuda()
    pos_loss = Variable(torch.zeros(1)).cuda()
    op_loss = Variable(torch.zeros(1)).cuda()
    if mode == 0:
        target_shapes, target_operations, target_sizes, target_positions = \
            labels

        # shape classification loss
        for i in range(time_steps):
            shape_loss += nllloss(out[0][i], target_shapes[:, i])

        # operation classification
        for i in range(time_steps):
            op_loss += nllloss(out[1][i], target_operations[:, i])

        # position regression
        for i in range(time_steps):
            for j in range(2):
                pos_loss += nllloss(out[3][i][j], target_positions[:, i, j])

        # size regression
        for i in range(time_steps):
            size_loss += nllloss(out[2][i], target_sizes[:, i])

    elif mode == 3 or mode == 4 or mode == 5:
        target_shapes, target_operations, target_sizes, target_positions, \
        target_which_type = labels
        which_op_loss = Variable(torch.zeros(1)).cuda()

        # shape classification loss
        for i in range(time_steps):
            shape_loss += torch.mean(
                neg_ll_loss(out[0][i], target_shapes[:, i],
                            (1 - target_which_type[:, i])))
        # operation classification
        for i in range(time_steps):
            op_loss += torch.mean(neg_ll_loss(out[1][i], target_operations[:,
                                                         i],
                                              target_which_type[:, i]))

            # position regression
        for i in range(time_steps):
            for j in range(2):
                pos_loss += torch.mean(
                    neg_ll_loss(out[3][i][j], target_positions[
                                              :, i, j],
                                (1 - target_which_type[:, i])))

        # size regression
        for i in range(time_steps):
            size_loss += torch.mean(neg_ll_loss(out[2][i], target_sizes[:, i],
                                                (1 - target_which_type[:, i])))

        # which_op loss (whether to "draw" or "op")
        for i in range(time_steps):
            which_op_loss += nllloss(out[4][i], target_which_type[:, i])

        # print(shape_loss.data.cpu().numpy(), pos_loss.data.cpu().numpy(),
        #       op_loss.data.cpu().numpy(), size_loss.data.cpu().numpy(),
        #       which_op_loss.data.cpu().numpy())
        return shape_loss + pos_loss + op_loss + size_loss + which_op_loss
    else:
        return shape_loss + pos_loss + op_loss + size_loss


def loss_from_set(outputs: List, labels_set: dict, time_steps: int):
    """
    In this type of training, we define a set of labels for every training 
    instance. These label from a set can all produce the target image that we 
    intend to make in the end. The idea is to let the network choose which 
    label it thinks it can produce. The idea is that, there are multiple 
    programs that can produce the same output, especially programs that have 
    and/or  kind of operations, you need to interchange the operands and you 
    still get the same target image.
    training instance.
    :param outputs: Outputs from the network
    :param time_steps: Time steps for which RNN is run
    :param labels_set: set of labels for every training instance
    :return: 
    """
    batch_size = outputs.size()[0]
    selected_labels = []
    for index in range(batch_size):
        loss_i = []
        # iterate over all the labels in the set for training instance "index"
        for l in labels_set[index]:
            loss_i.append(loss_one_instance(outputs, l, index, time_steps))
        # select the label which produces mini loss
        s_label = labels_set[torch.min(loss_i)[1]]
        selected_labels.append(s_label)
    selected_labels = torch.cat(selected_labels)

    # detach the selected labels from previous computation to prevent any non
    # differentiability creeping inside the computation graph
    selected_labels = selected_labels.detach()
    return losses_joint(outputs, selected_labels)


def loss_one_instance(outputs: List, label, index, time_steps):
    """
    Given one training instance, it calculates the loss
    :param outputs: outputs from the network
    :param index: index of the training instance, for which you want to 
    calculate the loss
    :param label: label of the corresponding to the training instance
    :param time_steps: time steps for which RNN is run
    :return: loss for that instance
    """
    loss = Variable(torch.zeros(1)).cuda()
    for t in range(time_steps):
        loss += nllloss(outputs[t][index:index + 1], label[:, t])
    return loss


def losses_joint(out, labels, time_steps, mode=0):
    """
    Defines losses for different modes
    :param out: output from the network
    :param labels: Ground truth labels
    :param time_steps: Length of the program
    :param mode: model type
    :return Loss: Sum of categoricam losses 
    """
    loss = Variable(torch.zeros(1)).cuda()

    for i in range(time_steps):
        loss += nllloss(out[i], labels[:, i])
    return loss


def neg_ll_loss(output, target, target_which_type):
    """
    Calculates the negative log likelihood loss and only returns a vector of 
    size batch_size
    :param output: predicted output from the network
    :param target: targets
    :return: 
    """
    batch_size = target.size()[0]
    loss = Variable(torch.zeros(batch_size)).cuda()
    for i in range(batch_size):
        loss[i] = -output[i, target.data[i]] * target_which_type.data[i]
    return loss
