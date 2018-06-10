""""
Contains small utility functions helpful in making the training interesting
"""
import h5py
import numpy as np
import torch
from torch.autograd.variable import Variable
from torch.utils.data import Dataset
from src.Models.models import ParseModelOutput
from typing import List


class Callbacks:
    """
    Callbacks to keep track of losses, acc etc during training
    """

    def __init__(self, batch_size, file_path):
        # elements to keep track
        self.elements = {}
        self.batch_size = batch_size
        self.file = file_path

    def add_element(self, elements):
        """
        Adds new element to be tracked
        :param elements: new variable that is needed to be tracked
        :return: 
        """
        for e in elements:
            self.elements[e] = []

    def add_value(self, element_value):
        """
        Adds value to the elment to be tracked
        :param element_value: Dict, mapping type of element to value to be added
        :return: 
        """
        for k in element_value.keys():
            self.elements[k].append(element_value[k])

    def dump_all(self):
        """
        dumps everything into a file
        :return: 
        """
        with h5py.File(self.file + ".hdf5", "w") as hf:
            for k in self.elements.keys():
                hf.create_dataset(data=np.array(self.elements[k]), name=k)


def pytorch_data(_generator, if_volatile=False):
    """Converts numpy tensor input data to pytorch tensors"""
    data_, labels = next(_generator)
    data = Variable(torch.from_numpy(data_))
    data.volatile = if_volatile
    data = data.cuda()
    labels = [Variable(torch.from_numpy(i)).cuda() for i in labels]
    return data, labels


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def prepare_input_op(arr, maxx):
    """
    This creates one-hot input for RNN that typically stores what happened
    in the immediate past. The first input to the RNN is
    start-of-the-sequence symbol. It is to be noted here that Input to the
    RNN in the form of one-hot contains one more element in comparison to
    the output from the RNN. This is because we don't want the
    start-of-the-sequence symbol in the output space of the program. arr
    here contains all the possible output that the RNN should/can produce,
    including stop-symbol. The stop symbol is represented by maxx-1 in the
    arr, but not to be bothered about here. Here, we make sure that the
    first input the RNN is start-of-the-sequence symbol by making maxx
    element of the array 1.
    :param arr: labels array
    :param maxx: maximum value in the labels
    :return: 
    """
    s = arr.shape
    array = np.zeros((s[0], s[1] + 1, maxx + 1), dtype=np.float32)
    # Start of sequence token.
    array[:, 0, maxx] = 1
    for i in range(s[0]):
        for j in range(s[1]):
            array[i, j + 1, arr[i, j]] = 1
    return array


def summary(model):
    """
    given the model, it returns a summary of learnable parameters
    :param model: Pytorch nn model
    :return: summary
    """
    state_dict = model.state_dict()
    total_param = 0
    num_parameters = {}
    for k in state_dict.keys():
        num_parameters[k] = np.prod([i for i in state_dict[k].size()])
        total_param += num_parameters[k]
    return num_parameters, total_param

def beams_parser(all_beams, batch_size, beam_width = 5):
    """
    Helper function that decodes and generates the expressions
    :param all_beams:
    :param batch_size:
    :param beam_width:
    :return:
    """
    all_expression = {}
    W = beam_width
    T = len(all_beams)
    for batch in range(batch_size):
        all_expression[batch] = []
        for w in range(W):
            temp = []
            parent = w
            for t in range(T - 1, -1, -1):
                temp.append(all_beams[t]["index"][batch, parent].data.cpu()
                            .numpy()[0])
                parent = all_beams[t]["parent"][batch, parent]
            temp = temp[::-1]
            all_expression[batch].append(np.array(temp))
        all_expression[batch] = np.squeeze(np.array(all_expression[batch]))
    return all_expression


def validity(program, max_time, timestep):
    """
    Checks the validity of the program.
    :param program: List of dictionary containing program type and elements
    :param max_time: Max allowed length of program
    :param timestep: Current timestep of the program, or in a sense length of 
    program
    # at evey index 
    :return: 
    """
    num_draws = 0
    num_ops = 0
    for i, p in enumerate(program):
        if p["type"] == "draw":
            # draw a shape on canvas kind of operation
            num_draws += 1
        elif p["type"] == "op":
            # +, *, - kind of operation
            num_ops += 1
        elif p["type"] == "stop":
            # Stop symbol, no need to process further
            if num_draws > ((len(program) - 1) // 2 + 1):
                return False
            if not (num_draws > num_ops):
                return False
            return (num_draws - 1) == num_ops

        if num_draws <= num_ops:
            # condition where number of operands are lesser than 2
            return False
        if num_draws > (max_time // 2 + 1):
            # condition for stack over flow
            return False
    if (max_time - 1) == timestep:
        if not ((num_draws - 1) == num_ops):
            return False
    return True

def stack_from_expressions(parser, expressions: List, downsample=None):
    """This take a generic expression as input and returns the final image for 
    this. The expressions need not be valid.
    :param parser: Object of the class parseModelOutput
    :param expression: List of expression
    :param downsample: factor by which to downsample voxel grid
    :return images: voxel representation of the expressions
    """
    stacks = []
    if downsample:
        meshgrid = np.arange(0, 64, downsample)
        xv, yv, zv = np.meshgrid(meshgrid, meshgrid, meshgrid, sparse=False, indexing='ij')

    for index, exp in enumerate(expressions):
        program = parser.sim.parse(exp)
        if validity(program, len(program), len(program) - 1):
            pass
        else:
            stack = np.zeros((parser.canvas_shape[0], parser.canvas_shape[1], parser.canvas_shape[2]))
            stacks.append(stack)
            continue
        # Use the primitives generated before.
        parser.sim.generate_stack(program, if_primitives=True)
        stack = parser.sim.stack_t
        stack = np.stack(stack, axis=0)[-1, 0, :, :]
        if downsample:
            stack = stack[xv, yv, zv]
        stacks.append(stack)
    stacks = np.stack(stacks, 0).astype(dtype=np.bool)
    return stacks



def voxels_from_expressions(expressions: List, primitives: dict, max_len=7, downsample=None):
    """This take a generic expression as input and returns the final voxel representation for
    this. The expressions need not be valid.
    :param expressions:
    :param primitives: dictionary, containg shape primitves in voxel grids, for faster
    processing. In general creating all shape primitives on-the-fly is an expensive operation.
    :param max_len: maximum length of programs
    :param downsample: factor by which to downsample voxel grid
    :return images: voxel representation of the expressions
    """
    stacks = []
    unique_draw = sorted(primitives.keys())

    parser = ParseModelOutput(unique_draw, max_len // 2 + 1, max_len, [64, 64, 64],
                              primitives=primitives)
    if downsample:
        meshgrid = np.arange(0, 64, downsample)
        xv, yv, zv = np.meshgrid(meshgrid, meshgrid, meshgrid, sparse=False, indexing='ij')

    for index, exp in enumerate(expressions):
        program = parser.sim.parse(exp)
        if not validity(program, len(program), len(program) - 1):
            stack = np.zeros((parser.canvas_shape[0], parser.canvas_shape[1], parser.canvas_shape[2]))
            stacks.append(stack)
            continue
        # Use the primitives generated before.
        parser.sim.generate_stack(program, if_primitives=True)
        stack = parser.sim.stack_t
        stack = np.stack(stack, axis=0)[-1, 0, :, :]
        if downsample:
            stack = stack[xv, yv, zv]
        stacks.append(stack)
    stacks = np.stack(stacks, 0).astype(dtype=np.bool)
    return stacks


class generator_iter(Dataset):
    """This is a helper function to be used in the parallel data loading using Pytorch
        DataLoader class"""

    def __init__(self, generator, train_size):
        self.generator = generator
        self.train_size = train_size

    def __len__(self):
        return self.train_size

    def __getitem__(self, idx):
        return next(self.generator)