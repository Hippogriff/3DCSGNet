# This script defines Pytorch models. We will test various NN architectures.
# From Teacher Forcing to the new stack-CNN, we will train them all in the
# similar fashion as we did in the 2D case.

import numpy as np
import torch
import torch.nn as nn
from torch.autograd.variable import Variable
from ..Generator.parser import Parser
from ..Generator.stack import SimulateStack


class CsgNet(nn.Module):
    def __init__(self,
                 grid_shape=[64, 64, 64],
                 dropout=0.5,
                 mode=1,
                 timesteps=3,
                 num_draws=400,
                 in_sz=2048,
                 hd_sz=2048,
                 stack_len=1):
        """
        This defines network architectures for CSG learning.
        :param dropout: Dropout to be used in non recurrent outputs of RNN
        :param mode: mode of training
        :param timesteps: Number of time steps in RNN
        :param num_draws: Number of unique primitives in the dataset
        :param in_sz: input size of features from encoder
        :param hd_sz: hidden size of RNN
        :param stack_len: Number of stack elements as input
        :param grid_shape: 3D grid structure.
        """
        super(CsgNet, self).__init__()

        self.input_channels = stack_len + 1
        self.in_sz = in_sz
        self.hd_sz = hd_sz
        self.num_draws = num_draws
        self.mode = mode
        self.time_steps = timesteps
        self.grid_shape = grid_shape
        self.rnn_layers = 1

        # Encoder architecture
        self.conv1 = nn.Conv3d(in_channels=self.input_channels, out_channels=32,
                               kernel_size=4, stride=(1, 1, 1), padding=(2,
                                                                         2, 2))
        self.b1 = nn.BatchNorm3d(num_features=32)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64,
                               kernel_size=4, stride=(1, 1, 1), padding=(2,
                                                                         2, 2))
        self.b2 = nn.BatchNorm3d(num_features=64)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128,
                               kernel_size=4, stride=(1, 1, 1), padding=(2,
                                                                         2, 2))
        self.b3 = nn.BatchNorm3d(num_features=128)
        self.conv4 = nn.Conv3d(in_channels=128, out_channels=256,
                               kernel_size=4, stride=(1, 1, 1), padding=(2,
                                                                         2, 2))
        self.b4 = nn.BatchNorm3d(num_features=256)
        self.conv5 = nn.Conv3d(in_channels=256, out_channels=256,
                               kernel_size=4, stride=(1, 1, 1), padding=(2,
                                                                         2, 2))
        self.b5 = nn.BatchNorm3d(num_features=256)

        # this sequential module is created for multi gpu training.
        self._encoder = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(dropout),
            self.b1,
            self.conv2,
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(dropout),
            self.b2,
            self.conv3,
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(dropout),
            self.b3,
            self.conv4,
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(dropout),
            self.b4,
            self.conv5,
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(dropout),
            self.b5,
        )

        # RNN architecture
        if (self.mode == 1) or (self.mode == 3):
            # Teacher forcing architecture, increased from previous value of 128
            self.input_op_sz = 128
            self.dense_input_op = nn.Linear(in_features=self.num_draws + 1,
                                            out_features=self.input_op_sz)

            self.rnn = nn.GRU(input_size=self.in_sz + self.input_op_sz,
                              hidden_size=self.hd_sz,
                              num_layers=self.rnn_layers,
                              batch_first=False)
            self.dense_output = nn.Linear(in_features=self.hd_sz, out_features=(
                self.num_draws))

        self.dense_fc_1 = nn.Linear(in_features=self.hd_sz,
                                    out_features=self.hd_sz)
        self.batchnorm_fc_1 = nn.BatchNorm1d(self.hd_sz, affine=False)

        self.pytorch_version = torch.__version__[2]
        if self.pytorch_version == "3":
            self.logsoftmax = nn.LogSoftmax(1)
        else:
            self.logsoftmax = nn.LogSoftmax()
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def encoder(self, x):
        """
        Only defines the forward pass of the encoder that takes the input 
        voxel Tensor and gives a fixed dim feature vector.
        :param x: Input tensor containing the raw voxels
        :return: fixed dim feature vector
        """
        batch_size = x.size()[0]
        x = self._encoder(x)
        x = x.view(1, batch_size, self.in_sz)
        return x

    def forward(self, x):
        """
        Defines the forward pass for the network
        :param x: This will contain data based on the type of training that 
        you do.
        :return: outputs of the network, depending upon the architecture 
        """
        if self.mode == 1:
            """Teacher forcing network"""
            data, input_op, program_len = x
            data = data.permute(1, 0, 2, 3, 4, 5)
            batch_size = data.size()[1]
            h = Variable(torch.zeros(1, batch_size, self.hd_sz)).cuda()
            x_f = self.encoder(data[-1, :, 0:1, :, :, :])
            outputs = []
            for timestep in range(0, program_len + 1):
                # X_f is always input to the network at every time step
                # along with previous predicted label
                input_op_rnn = self.relu(self.dense_input_op(input_op[:,
                                                             timestep, :]))
                input_op_rnn = input_op_rnn.unsqueeze(0)
                input = torch.cat((self.drop(x_f), input_op_rnn), 2)
                h, _ = self.rnn(input, h)
                hd = self.relu(self.dense_fc_1(self.drop(h[0])))
                output = self.logsoftmax(self.dense_output(self.drop(hd)), )
                outputs.append(output)
            return outputs

    def test(self, data):
        """ Describes test behaviour of different models"""
        if self.mode == 1:
            """Testing for teacher forcing"""
            data, input_op, program_len = data

            # This permute is used for multi gpu training, where first dimension is
            # considered as batch dimension.
            data = data.permute(1, 0, 2, 3, 4, 5)
            batch_size = data.size()[1]
            h = Variable(torch.zeros(1, batch_size, self.hd_sz)).cuda()
            x_f = self.encoder(data[-1, :, 0:1, :, :, :])
            last_op = input_op[:, 0, :]
            outputs = []
            for timestep in range(0, program_len):
                # X_f is always input to the network at every time step
                # along with previous predicted label
                input_op_rnn = self.relu(self.dense_input_op(last_op))
                input_op_rnn = input_op_rnn.unsqueeze(0)
                input = torch.cat((self.drop(x_f), input_op_rnn), 2)
                h, _ = self.rnn(input, h)
                hd = self.relu(self.dense_fc_1(self.drop(h[0])))
                output = self.logsoftmax(self.dense_output(self.drop(hd)))
                outputs.append(output)
                next_input_op = torch.max(output, 1)[1]
                arr = Variable(torch.zeros(batch_size, self.num_draws + 1).scatter_(1,
                                                                                    next_input_op.data.cpu().view(batch_size, 1),
                                                                                    1.0)).cuda()
                last_op = arr
            return outputs

    def beam_search_mode_1(self, data, w, max_time):
        """
        Implements beam search for different models.
        :param x: Input data
        :param w: beam width
        :param max_time: Maximum length till the program has to be generated
        :return all_beams: all beams to find out the indices of all the 
        """
        data, input_op = data

        # Beam, dictionary, with elements as list. Each element of list
        # containing index of the selected output and the corresponding
        # probability.
        data = data.permute(1, 0, 2, 3, 4, 5)
        pytorch_version = torch.__version__[2]
        batch_size = data.size()[1]
        h = Variable(torch.zeros(1, batch_size, self.hd_sz)).cuda()
        # Last beams' data
        B = {0: {"input": input_op, "h": h}, 1: None}
        next_B = {}
        x_f = self.encoder(data[-1, :, 0:1, :, :, :])
        prev_output_prob = [Variable(torch.ones(batch_size, self.num_draws)).cuda()]
        all_beams = []
        all_inputs = []
        stopped_programs = np.zeros((batch_size, w), dtype=bool)
        for timestep in range(0, max_time):
            outputs = []
            for b in range(w):
                if not B[b]:
                    break
                input_op = B[b]["input"]

                h = B[b]["h"]
                input_op_rnn = self.relu(self.dense_input_op(input_op[:, 0, :]))
                input_op_rnn = input_op_rnn.view(1, batch_size,
                                                 self.input_op_sz)
                input = torch.cat((x_f, input_op_rnn), 2)
                h, _ = self.rnn(input, h)
                hd = self.relu(self.dense_fc_1(self.drop(h[0])))
                dense_output = self.dense_output(self.drop(hd))
                output = self.logsoftmax(dense_output)
                # Element wise multiply by previous probabs
                if pytorch_version == "3":
                    output = torch.nn.Softmax(1)(output)
                elif pytorch_version == "1":
                    output = torch.nn.Softmax()(output)
                output = output * prev_output_prob[b]
                outputs.append(output)
                next_B[b] = {}
                next_B[b]["h"] = h

            if len(outputs) == 1:
                outputs = outputs[0]
            else:
                outputs = torch.cat(outputs, 1)

            next_beams_index = torch.topk(outputs, w, 1, sorted=True)[1]
            next_beams_prob = torch.topk(outputs, w, 1, sorted=True)[0]
            # print (next_beams_prob)
            current_beams = {"parent": next_beams_index.data.cpu().numpy() // (
                self.num_draws),
                             "index": next_beams_index % (self.num_draws)}
            # print (next_beams_index % (self.num_draws))
            next_beams_index %= (self.num_draws)
            all_beams.append(current_beams)

            # Update previous output probabilities
            temp = Variable(torch.zeros(batch_size, 1)).cuda()
            prev_output_prob = []
            for i in range(w):
                for index in range(batch_size):
                    temp[index, 0] = next_beams_prob[index, i]
                prev_output_prob.append(temp.repeat(1, self.num_draws))
            # hidden state for next step
            B = {}
            for i in range(w):
                B[i] = {}
                temp = Variable(torch.zeros(h.size())).cuda()
                for j in range(batch_size):
                    temp[0, j, :] = next_B[current_beams["parent"][j, i]]["h"][0, j, :]
                B[i]["h"] = temp

            # one_hot for input to the next step
            for i in range(w):
                arr = Variable(torch.zeros(batch_size, self.num_draws + 1)
                               .scatter_(1, next_beams_index[:, i:i + 1].data.cpu(),
                                         1.0)).cuda()
                B[i]["input"] = arr.unsqueeze(1)
            all_inputs.append(B)

        return all_beams, next_beams_prob, all_inputs


class ParseModelOutput:
    """
    This class parse complete output from the network which are in joint
    fashion. This class can be used to generate final canvas and
    expressions.
    """
    def __init__(self, unique_draws, stack_size, steps, canvas_shape, primitives=None):
        """
        :param unique_draws: Unique draw/op operations in the current dataset
        :param stack_size: Stack size
        :param steps: Number of steps in the program
        :param canvas_shape: Shape of the canvases
        :param primitives: dictionary containing 3D voxel representation of primitives
        """
        self.canvas_shape = canvas_shape
        self.stack_size = stack_size
        self.steps = steps
        self.Parser = Parser()
        if primitives == None:
            self.sim = SimulateStack(self.stack_size, self.canvas_shape, unique_draws)
        else:
            self.sim = SimulateStack(self.stack_size, self.canvas_shape, unique_draws)
            self.sim.get_all_primitives(primitives)
        self.unique_draws = unique_draws

    def get_final_canvas(self, outputs, if_just_expressions=False,
                         if_pred_images=False):
        """
        Takes the raw output from the network and returns the predicted 
        canvas. The steps involve parsing the outputs into expressions, 
        decoding expressions, and finally producing the canvas using 
        intermediate stacks.
        :type if_pred_images: bool, either use it with fixed len programs or keep it 
        True, because False doesn't work with variable length programs
        :param if_just_expressions: If only expression is required than we 
        just return the function after calculating expressions
        :param outputs: List, each element correspond to the output from the 
        network
        :return: stack: Predicted final stack for correct programs
        :return: correct_programs: Indices of correct programs
        """
        pytorch_version = torch.__version__[2]
        batch_size = outputs[0].size()[0]
        steps = self.steps
        # Initialize empty expression string, len equal to batch_size
        correct_programs = []
        expressions = [""] * batch_size
        labels = [torch.max(o, 1)[1].data.cpu().numpy() for o in outputs]

        for j in range(batch_size):
            for i in range(steps):
                if pytorch_version == "3":
                    expressions[j] += self.unique_draws[labels[i][j]]
                elif pytorch_version == "1":
                    expressions[j] += self.unique_draws[labels[i][j, 0]]
        # Remove the stop symbol and later part of the expression
        for index, exp in enumerate(expressions):
            expressions[index] = exp.split("$")[0]

        if if_just_expressions:
            return expressions
        stacks = []
        for index, exp in enumerate(expressions):
            program = self.sim.parse(exp)
            if validity(program, len(program), len(program) - 1):
                correct_programs.append(index)
            else:
                if if_pred_images:
                    # if you just want final predicted image
                    stack = np.zeros((self.canvas_shape[0],
                                      self.canvas_shape[1],
                                      self.canvas_shape[2]))
                else:
                    stack = np.zeros((self.steps + 1, self.stack_size,
                                      self.canvas_shape[0],
                                      self.canvas_shape[1],
                                      self.canvas_shape[2]))
                stacks.append(stack)
                continue
                # Check the validity of the expressions

            self.sim.generate_stack(program, if_primitives=True)
            stack = self.sim.stack_t
            stack = np.stack(stack, axis=0)
            if if_pred_images:
                stacks.append(stack[-1, 0, :, :, :])
            else:
                stacks.append(stack)
        if if_pred_images:
            stacks = np.stack(stacks, 0).astype(dtype=np.bool)
        else:
            stacks = np.stack(stacks, 1).astype(dtype=np.bool)
        return stacks, correct_programs, expressions

    def expression2stack(self, expressions):
        """Assuming all the expression are correct and coming from 
        groundtruth labels. Helpful in visualization of programs
        :param expressions: List, each element an expression of program
        """
        stacks = []
        for index, exp in enumerate(expressions):
            program = self.sim.parse(exp)
            self.sim.generate_stack(program, if_primitives=True)
            stack = self.sim.stack_t
            stack = np.stack(stack, axis=0)
            stacks.append(stack)
        stacks = np.stack(stacks, 1).astype(dtype=np.float32)
        return stacks

    def labels2exps(self, labels, steps):
        """
        Assuming grountruth labels, we want to find expressions for them
        :param labels: Grounth labels batch_size x time_steps
        :return: expressions: Expressions corresponding to labels
        """
        if isinstance(labels, np.ndarray):
            batch_size = labels.shape[0]
        else:
            batch_size = labels.size()[0]
            labels = labels.data.cpu().numpy()
        # Initialize empty expression string, len equal to batch_size
        correct_programs = []
        expressions = [""] * batch_size
        for j in range(batch_size):
            for i in range(steps):
                expressions[j] += self.unique_draws[labels[j, i]]
        return expressions


class PushDownInduceProgram:
    def __init__(self, stack_size, canvas_shape, unique_draws, max_time=3,
                 batch_size=256, primitves=None):
        """
        Parses the output from the network one time step at a time and also 
        simulate the stack that becomes input to the network at the next time
        step. It uses the pushdown stack created for Stack-CNN. 
        This class is used in testing a stack based model or in training and 
        testing of RL model
        :param stack_size: Max size of the stack
        :param canvas_shape: Shape of the canvas drawing
        :param max_time: Number of time steps that a program can run
        :param batch_size: Size of mini batch for training
        """
        self.stack_size = stack_size
        self.batch_size = batch_size
        self.unique_draws = unique_draws
        self.canvas_shape = canvas_shape
        # for wrong programs
        self.wrong_indices = np.zeros(self.batch_size, dtype=bool)

        # Programs that have produced stop symbol
        self.stopped_programs = np.zeros(self.batch_size, dtype=bool)
        self.max_time = max_time
        # stores the intermediate stack state.
        self.stack = np.zeros((self.batch_size, stack_size, canvas_shape[0],
                               canvas_shape[1], canvas_shape[2]), dtype=int)
        self.Parser = Parser()
        # create stack simulator for every batch element so that you can
        # resume simulation when new instruction arrives
        self.sim = [
            SimulateStack(self.stack_size, self.canvas_shape, unique_draws) for
            _ in range(self.batch_size)]
        if primitves:
            for s in self.sim:
                s.get_all_primitives(primitves)

        self.expressions = [""] * self.batch_size

    def induce_program(self, output, timestep):
        """
        Induces the program by taking current output from the network, 
        returns the simulated stack. Also takes care of the validity of 
        the programs. Currently, as soon as the program is recognized to be 
        wrong, it just drops it.
        For invalid programs we produce empty canvas.
        For programs that stop and are valid, the state of the canvas at the 
        stopped timestep is repeated.
        If we encounter any wrong instruction, we lose hope and forget it.
        :param output: Output from the network at some point of time
        :param timestep: Current time step at which output is produced
        :return: stack: simulated stack at that time step
        """
        # Get the current expression
        label = torch.max(output, 1)[1].data.cpu().numpy()

        for index in range(self.batch_size):
            exp = self.unique_draws[label[index]]
            self.expressions[index] += exp
            program = self.tokenizer(self.expressions[index])

            if self.wrong_indices[index] or self.stopped_programs[index]:
                # if it was already wrong program before or is stopped,
                # continue to next training instance
                continue

            if (exp == "$"):
                # program has stopped and wasn't previously stopped
                self.wrong_indices[index] = (not validity(program,
                                                          self.max_time,
                                                          timestep))

                self.stopped_programs[index] = True
                if self.wrong_indices[index]:
                    # prorgam invalid, output zero canvas (it will make
                    # debugging difficult for the wrong programs)
                    self.stack[index, :, :, :, :] = np.zeros(
                        (self.stack_size, self.canvas_shape[0],
                         self.canvas_shape[1], self.canvas_shape[2]))
                continue

            self.wrong_indices[index] = (not validity(program, self.max_time,
                                                      timestep))

            if (not self.wrong_indices[index]):
                # Continue simulating stack from last instruction
                temp = [program[-1]]
                self.sim[index].generate_stack(temp, start_scratch=False, if_primitives=True)
                # take only the last time step stack and reject others
                self.stack[index, :, :, :, :] = self.sim[index].stack_t[-1]
            elif self.wrong_indices[index]:
                # prorgam invalid, output zero canvas (it will make debugging
                #  difficult for the wrong programs)
                self.stack[index, :, :, :, :] = np.zeros(
                    (self.stack_size, self.canvas_shape[0],
                     self.canvas_shape[1], self.canvas_shape[2]))

        stack = np.expand_dims(self.stack.astype(np.float32), 0)
        return Variable(torch.from_numpy(stack)).cuda(), self.expressions

    def tokenizer(self, expression):
        """
        NOTE: This method is different from parse method in Parser class
        Takes an expression, extracts tokens
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
            elif value == "$":
                # must be a stop symbol
                program.append({})
                program[-1]["type"] = "stop"
                program[-1]["value"] = "$"
        return program


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
