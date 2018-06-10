"""
Generates training and testing data in mini batches.
"""

import deepdish as dd
import numpy as np
from matplotlib import pyplot as plt
from .parser import Parser
from .stack import SimulateStack


class Generator:
    """
    Primary function of this generator is to generate variable length
    dataset for training variable length programs. It creates a generator
    object for every length of program that you want to generate. This
    process allow finer control of every batch that you feed into the
    network. This class can also be used in fixed length training.
    """
    def __init__(self,
                 data_labels_paths,
                 batch_size=32,
                 time_steps=3,
                 stack_size=2,
                 canvas_shape=[64, 64, 64],
                 primitives=None):
        """
        :param stack_size: maximum size of stack used for programs.
        :param canvas_shape: canvas shape
        :param primitives: Dictionary containing pre-rendered shape primitives in the
         form of grids.
        :param time_steps: Max time steps for generated programs
        :param data_labels_paths: dictionary containing paths for different
        lengths programs
        :param batch_size: batch_size
        """
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.canvas_shape = canvas_shape
        self.stack_size = stack_size

        self.programs = {}
        self.data_labels_path = data_labels_paths
        for index in data_labels_paths.keys():
            with open(data_labels_paths[index]) as data_file:
                self.programs[index] = data_file.readlines()
        all_programs = []

        for k in self.programs.keys():
            all_programs += self.programs[k]

        self.unique_draw = self.get_draw_set(all_programs)
        self.unique_draw.sort()
        # Append ops in the end and the last one is for stop symbol
        self.unique_draw += ["+", "*", "-", "$"]
        sim = SimulateStack(self.time_steps // 2 + 1, self.canvas_shape,
                            self.unique_draw)

        if not (type(primitives) is dict):
            # # Draw all primitive in one go and reuse them later
            # sim.draw_all_primitives(self.unique_draw)
            # self.primitives = sim.draw_all_primitives(self.unique_draw)
            # dd.io.save("mix_len_all_primitives.h5", self.primitives)
            self.primitives = dd.io.load('data/primitives.h5')
        else:
            self.primitives = primitives
        self.parser = Parser()

    def parse(self, expression):
        """
        NOTE: This method is different from parse method in Parser class
        Takes an expression, returns a serial program
        :param expression: program expression in postfix notation
        :return program:
        """
        self.shape_types = ["u", "p", "y"]
        self.op = ["*", "+", "-"]
        program = []
        for index, value in enumerate(expression):
            if value in self.shape_types:
                program.append({})
                program[-1]["type"] = "draw"

                # find where the parenthesis closes
                close_paren = expression[index:].index(")") + index
                program[-1]["value"] = expression[index:close_paren + 1]
            elif value in self.op:
                program.append({})
                program[-1]["type"] = "op"
                program[-1]["value"] = value
            else:
                pass
        return program

    def get_draw_set(self, expressions):
        """
        Find a sorted set of draw type from the entire dataset. The idea is to
        use only the plausible position, scale and shape combinations and
        reject that are not possible because of the restrictions we have in
        the dataset.
        :param expressions: List containing entire dataset in the form of
        expressions.
        :return: unique_chunks: Unique sorted draw operations in the dataset.
        """
        shapes = ["u", "p", "y"]
        chunks = []
        for expression in expressions:
            for i, e in enumerate(expression):
                if e in shapes:
                    index = i
                    last_index = expression[index:].index(")")
                    chunks.append(expression[index:index + last_index + 1])
        return list(set(chunks))

    def get_train_data(self, batch_size: int, program_len: int,
                       final_canvas=False, if_randomize=True, if_primitives=False,
                       num_train_images=400, if_jitter=False):
        """
        This is a special generator that can generate dataset for any length.
        This essentially corresponds to the "variable len program"
        experiment. Here, generate a dataset for training for fixed length.
        Since, this is a generator, you need to make a generator object for
        all different kind of lengths and use them as required. It is made
        sure that samples are shuffled only once in an epoch and all the
        samples are different in an epoch.
        :param if_randomize: whether to randomize the training instance during training.
        :param if_primitives: if pre-rendered primitives are given
        :param num_train_images: Number of training instances
        :param if_jitter: whether to jitter the voxels or not
        :param batch_size: batch size for the current program
        :param program_len: which program length dataset to sample
        :param final_canvas: This is special mode of data generation where
        all the dataset is loaded in one go and iteratively yielded. The
        dataset for only target images is created.
        """
        # The last label corresponds to the stop symbol and the first one to
        # start symbol.
        labels = np.zeros((batch_size, program_len + 1), dtype=np.int64)
        sim = SimulateStack(program_len // 2 + 1, self.canvas_shape,
                            self.unique_draw)
        sim.get_all_primitives(self.primitives)
        parser = Parser()

        if final_canvas:
            # We will load all the final canvases from the disk.
            path = self.data_labels_path[program_len]
            path = path[0:-15]
            Stack = np.zeros((1, num_train_images, 1, self.canvas_shape[0],
                              self.canvas_shape[1], self.canvas_shape[2]),
                             dtype=np.bool)
            for i in range(num_train_images):
                p = path + "{}.png".format(i + 1)
                img = plt.imread(p)[:, :, 0]
                Stack[0, i, 0, :, :] = img.astype(np.bool)

        while True:
            # Random things to select random indices
            IDS = np.arange(num_train_images)
            if if_randomize:
                np.random.shuffle(IDS)
            for rand_id in range(0, num_train_images - batch_size,
                                 batch_size):
                image_ids = IDS[rand_id:rand_id + batch_size]
                if not final_canvas:
                    stacks = []
                    sampled_exps = []
                    for index, value in enumerate(image_ids):
                        sampled_exps.append(self.programs[program_len][value])
                        if not if_primitives:
                            program = parser.parse(
                                self.programs[program_len][value])
                        else:
                            # if all primitives are give already, parse using
                            #  different parser to get the keys to dict
                            program = self.parse(self.programs[program_len][
                                                         value])
                        sim.generate_stack(program, if_primitives=if_primitives)
                        stack = sim.stack_t
                        stack = np.stack(stack, axis=0)
                        stacks.append(stack)
                    stacks = np.stack(stacks, 1).astype(dtype=np.float32)
                else:
                    # When only target image is required
                    stacks = Stack[0:1, image_ids, 0:1, :, :, :].astype(
                        dtype=np.float32)
                for index, value in enumerate(image_ids):
                    # Get the current program
                    exp = self.programs[program_len][value]
                    program = self.parse(exp)
                    for j in range(program_len):
                        labels[index, j] = self.unique_draw.index(
                            program[j]["value"])

                    labels[:, -1] = len(self.unique_draw) - 1

                if if_jitter:
                    temp = stacks[-1, :, 0, :, :, :]
                    stacks[-1, :, 0, :, :, :] = np.roll(temp, (np.random.randint(-3, 4),
                                                               np.random.randint(-3, 4),
                                                               np.random.randint(-3, 4)),
                                                        axis=(1, 2, 3))

                yield [stacks, labels]

    def get_test_data(self, batch_size: int, program_len: int,
                      if_randomize=False, final_canvas=False,
                      num_train_images=None, num_test_images=None,
                      if_primitives=False, if_jitter=False):
        """
        Test dataset creation. It is assumed that the first num_training
        examples in the dataset corresponds to training and later num_test
        are validation dataset. The validation may optionally be shuffled
        randomly but usually not required.
        :param num_train_images:
        :param if_primitives: if pre-rendered primitives are given
        :param if_jitter: Whether to jitter the voxel grids
        :param num_test_images: Number of test images
        :param batch_size: batch size of dataset to yielded
        :param program_len: length of program to be generated
        :param if_randomize: if randomize
        :param final_canvas: if true return only the target canvas instead of 
        complete stack to save memory
        :return: 
        """
        # This generates test data of fixed length. Samples are not shuffled
        # by default.
        labels = np.zeros((batch_size, program_len + 1), dtype=np.int64)
        sim = SimulateStack(program_len // 2 + 1, self.canvas_shape,
                            self.unique_draw)
        sim.get_all_primitives(self.primitives)
        parser = Parser()

        if final_canvas:
            # We will load all the final canvases from the disk.
            path = self.data_labels_path[program_len]
            path = path[0:-15]
            Stack = np.zeros((1, num_test_images, 1, self.canvas_shape[0],
                              self.canvas_shape[1], self.canvas_shape[2]),
                             dtype=np.bool)
            for i in range(num_train_images,
                           num_test_images + num_train_images):
                p = path + "{}.png".format(i + 1)
                img = plt.imread(p)[:, :, 0]
                Stack[0, i, 0, :, :] = img.astype(np.bool)

        while True:
            # Random things to select random indices
            IDS = np.arange(num_train_images, num_train_images +
                            num_test_images)
            if if_randomize:
                np.random.shuffle(IDS)
            for rand_id in range(0, num_test_images - batch_size, batch_size):
                image_ids = IDS[rand_id: rand_id + batch_size]
                if not final_canvas:
                    stacks = []
                    sampled_exps = []
                    for index, value in enumerate(image_ids):
                        sampled_exps.append(self.programs[program_len][value])
                        if not if_primitives:
                            program = parser.parse(
                                self.programs[program_len][value])
                        if True:
                            # if all primitives are give already, parse using
                            #  different parser to get the keys to dict
                            try:
                                program = self.parse(self.programs[program_len][
                                                         value])
                            except:
                                print(index, self.programs[program_len][
                                    value])
                        sim.generate_stack(program, if_primitives=if_primitives)
                        stack = sim.stack_t
                        stack = np.stack(stack, axis=0)
                        stacks.append(stack)
                    stacks = np.stack(stacks, 1).astype(dtype=np.float32)
                else:
                    # When only target image is required
                    stacks = Stack[0:1, image_ids, 0:1, :, :, :].astype(
                        dtype=np.float32)
                for index, value in enumerate(image_ids):
                    # Get the current program
                    exp = self.programs[program_len][value]
                    program = self.parse(exp)
                    for j in range(program_len):
                        try:
                            labels[index, j] = self.unique_draw.index(
                                program[j]["value"])
                        except:
                            print(program)

                    labels[:, -1] = len(self.unique_draw) - 1

                if if_jitter:
                    temp = stacks[-1, :, 0, :, :, :]
                    stacks[-1, :, 0, :, :, :] = np.roll(temp, (np.random.randint(-3, 4),
                                                               np.random.randint(-3, 4),
                                                               np.random.randint(-3, 4)),
                                                        axis=(1, 2, 3))
                yield [stacks, labels]
