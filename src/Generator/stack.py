"""
This constructs stack from the expressions. This is specifically tailored for 3D
CSG. Most of the ideas are taken from our previous work on 2D CSG.
"""

import numpy as np
from .parser import Parser

class PushDownStack(object):
    """Simple PushDown Stack implements in the form of array"""

    def __init__(self, max_len, canvas_shape):
        _shape = [max_len] + canvas_shape
        self.max_len = max_len
        self.canvas_shape = canvas_shape
        self.items = []
        self.max_len = max_len

    def push(self, item):
        if len(self.items) >= self.max_len:
            assert False, "exceeds max len for stack!!"
        self.items = [item.copy()] + self.items

    def pop(self):
        if len(self.items) == 0:
            assert False, "below min len of stack!!"
        item = self.items[0]
        self.items = self.items[1:]
        return item

    def get_items(self):
        """
        In this we create a fixed shape tensor amenable for further usage
        :return:
        """
        size = [self.max_len] + self.canvas_shape
        stack_elements = np.zeros(size, dtype=bool)
        length = len(self.items)
        for j in range(length):
            stack_elements[j, :, :, :] = self.items[j]
        return stack_elements

    def clear(self):
        """Re-initializes the stack"""
        self.items = []


class SimulateStack:
    """
    Simulates the stack for CSG
    """
    def __init__(self, max_len, canvas_shape, draw_uniques):
        """
        :param max_len: max size of stack
        :param canvas_shape: canvas shape
        :param draw_uniques: unique operations (draw + ops)
        """
        self.draw_obj = Draw(canvas_shape=canvas_shape)
        self.draw = {
            "u": self.draw_obj.draw_cube,
            "p": self.draw_obj.draw_sphere,
            "y": self.draw_obj.draw_cylinder
        }
        self.op = {"*": self._and, "+": self._union, "-": self._diff}
        self.stack = PushDownStack(max_len, canvas_shape)
        self.stack_t = []
        self.stack.clear()
        self.stack_t.append(self.stack.get_items())
        self.parser = Parser()

    def draw_all_primitives(self, draw_uniques):
        """
        Draws all primitives so that we don't have to draw them over and over.
        :param draw_uniques: unique operations (draw + ops)
        :return:
        """
        self.primitives = {}
        for index, value in enumerate(draw_uniques[0:-4]):
            p = self.parser.parse(value)[0]
            which_draw = p["value"]
            if which_draw == "u" or which_draw == "p":
                # draw cube or sphere
                x = int(p["param"][0])
                y = int(p["param"][1])
                z = int(p["param"][2])
                radius = int(p["param"][3])
                layer = self.draw[which_draw]([x, y, z], radius)

            elif which_draw == "y":
                # draw cylinder
                # TODO check if the order is correct.
                x = int(p["param"][0])
                y = int(p["param"][1])
                z = int(p["param"][2])
                radius = int(p["param"][3])
                height = int(p["param"][4])
                layer = self.draw[p["value"]]([x, y, z], radius, height)
            self.primitives[value] = layer
        return self.primitives

    def get_all_primitives(self, primitives):
        """ Get all primitive from outseide class
        :param primitives: dictionary containing pre-rendered shape primitives
        """
        self.primitives = primitives

    def parse(self, expression):
        """
        NOTE: This method generates terminal symbol for an input program expressions.
        :param expression: program expression in postfix notation
        :return program:
        """
        shape_types = ["u", "p", "y"]
        op = ["*", "+", "-"]
        program = []
        for index, value in enumerate(expression):
            if value in shape_types:
                program.append({})
                program[-1]["type"] = "draw"

                # find where the parenthesis closes
                close_paren = expression[index:].index(")") + index
                program[-1]["value"] = expression[index:close_paren + 1]
            elif value in op:
                program.append({})
                program[-1]["type"] = "op"
                program[-1]["value"] = value
            else:
                pass
        return program

    def generate_stack(self, program: list, start_scratch=True, if_primitives=False):
        """
        Executes the program step-by-step and stores all intermediate stack
        states.
        :param if_primitives: if pre-rendered primitives are given.
        :param program: List with each item a program step
        :param start_scratch: whether to start creating stack from scratch or 
        stack already exist and we are appending new instructions. With this 
        set to False, stack can be started from its previous state.
        """
        # clear old garbage
        if start_scratch:
            self.stack_t = []
            self.stack.clear()
            self.stack_t.append(self.stack.get_items())

        for index, p in enumerate(program):
            if p["type"] == "draw":
                if if_primitives:
                    # fast retrieval of shape primitive
                    layer = self.primitives[p["value"]]
                    self.stack.push(layer)
                    self.stack_t.append(self.stack.get_items())
                    continue

                if p["value"] == "u" or p["value"] == "p":
                    # draw cube or sphere
                    x = int(p["param"][0])
                    y = int(p["param"][1])
                    z = int(p["param"][2])
                    radius = int(p["param"][3])
                    layer = self.draw[p["value"]]([x, y, z], radius)

                elif p["value"] == "y":
                    # draw cylinder
                    # TODO check if the order is correct.
                    x = int(p["param"][0])
                    y = int(p["param"][1])
                    z = int(p["param"][2])
                    radius = int(p["param"][3])
                    height = int(p["param"][4])
                    layer = self.draw[p["value"]]([x, y, z], radius, height)
                self.stack.push(layer)

                # Copy to avoid orver-write
                # self.stack_t.append(self.stack.items.copy())
                self.stack_t.append(self.stack.get_items())
            else:
                # operate
                obj_2 = self.stack.pop()
                obj_1 = self.stack.pop()
                layer = self.op[p["value"]](obj_1, obj_2)
                self.stack.push(layer)
                # Copy to avoid over-write
                # self.stack_t.append(self.stack.items.copy())
                self.stack_t.append(self.stack.get_items())

    def _union(self, obj1, obj2):
        """Union between voxel grids"""
        return np.logical_or(obj1, obj2)

    def _and(self, obj1, obj2):
        """Intersection between voxel grids"""
        return np.logical_and(obj1, obj2)

    def _diff(self, obj1, obj2):
        """Subtraction between voxel grids"""
        return (obj1 * 1. - np.logical_and(obj1, obj2) * 1.).astype(np.bool)


class Draw:
    def __init__(self, canvas_shape=[64, 64, 64]):
        """
        Helper Class for drawing the canvases.
        :param canvas_shape: shape of the canvas on which to draw objects
        """
        self.canvas_shape = canvas_shape

    def draw_sphere(self, center, radius):
        """Makes sphere inside a cube of canvas_shape
        :param center: center of the sphere
        :param radius: radius of sphere
        :return:
        """
        radius -= 1
        canvas = np.zeros(self.canvas_shape, dtype=bool)
        for x in range(center[0] - radius, center[0] + radius + 1):
            for y in range(center[1] - radius, center[1] + radius + 1):
                for z in range(center[2] - radius, center[2] + radius + 1):
                    if np.linalg.norm(np.array(center) - np.array(
                            [x, y, z])) <= radius:
                        canvas[x, y, z] = True
        return canvas

    def draw_cube(self, center, side):
        """Makes cube inside a cube of canvas_shape
        :param center: center of cube
        :param side: side of cube
        :return:
        """
        side -= 1
        canvas = np.zeros(self.canvas_shape, dtype=bool)
        side = side // 2
        for x in range(center[0] - side, center[0] + side + 1):
            for y in range(center[1] - side, center[1] + side + 1):
                for z in range(center[2] - side, center[2] + side + 1):
                    canvas[x, y, z] = True
        return canvas

    def draw_cylinder(self, center, radius, height):
        """Makes cylinder inside a of canvas_shape
        :param center: center of cylinder
        :param radius: radius of cylinder
        :param height: height of cylinder
        :return:
        """
        radius -= 1
        height -= 1
        canvas = np.zeros(self.canvas_shape, dtype=bool)

        for z in range(center[2] - int(height / 2),
                       center[2] + int(height / 2) + 1):
            for x in range(center[0] - radius, center[0] + radius + 1):
                for y in range(center[1] - radius, center[1] + radius + 1):
                    if np.linalg.norm(
                                    np.array([center[0], center[1]]) - np.array(
                                [x, y])) <= radius:
                        canvas[x, y, z] = True
        return canvas