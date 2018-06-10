"""
A crude parser, specific to the 3D CSG dataset
sample of an expression: cu(16,40,32,28)cu(16,40,32,28)cu(16,40,32,28)cu(16,
40,32,28)cu(24,32,24,32)-
"""
import string


class Parser:
    """
    Parser to parse the program written in postfix notation
    """

    def __init__(self):
        self.shape_types = ["u", "p", "y"]
        self.op = ["*", "+", "-"]

    def parse(self, expression: string):
        """
        Takes an empression, returns a serial program
        :param expression: program expression in postfix notation
        :return program:
        """
        program = []
        for index, value in enumerate(expression):
            if value in self.shape_types:
                # draw shape instruction
                program.append({})
                program[-1]["value"] = value
                program[-1]["type"] = "draw"
                # find where the parenthesis closes
                close_paren = expression[index:].index(")") + index

                program[-1]["param"] = expression[index + 2:close_paren].split(
                    ",")
                if program[-1]["param"][0][0] == "(":
                    print (expression)

            elif value in self.op:
                # operations instruction
                program.append({})
                program[-1]["type"] = "op"
                program[-1]["value"] = value

            elif value == "$":
                # must be a stop symbol
                program.append({})
                program[-1]["type"] = "stop"
                program[-1]["value"] = "$"
        return program
