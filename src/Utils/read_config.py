from configobj import ConfigObj


class Config(object):
    "Read from a config file"

    def __init__(self, filename, if_gen=False):
        self.filename = filename
        config = ConfigObj(self.filename)
        self.config = config

        # meaningful comment about the experiment
        self.comment = config["comment"]

        # name of the model to be trained
        self.model_path = config["train"]["model_path"]

        # pretrained model flag
        self.preload_model = config["train"].as_bool("preload_model")

        # path of the pretrained model
        self.pretrain_modelpath = config["train"]["pretrain_model_path"]

        # proportion of the dataset to train or test on
        self.proportion = config["train"].as_int("proportion")

        # Number of times gradients need to be accumulated before updating the weights
        self.num_traj = config["train"].as_int("num_traj")

        # number of epochs to train
        self.epochs = config["train"].as_int("num_epochs")

        # batch size for training
        self.batch_size = config["train"].as_int("batch_size")

        # hidden size for RNN
        self.hidden_size = config["train"].as_int("hidden_size")

        # image feature size
        self.input_size = config["train"].as_int("input_size")

        # mode of training, not applicable
        self.mode = config["train"].as_int("mode")

        # Learning rate
        self.lr = config["train"].as_float("lr")

        # weight decay
        self.weight_decay = config["train"].as_float("weight_decay")

        # dropout
        self.dropout = config["train"].as_float("dropout")

        # number of epochs to wait before the learning rate is to be reduced up being flat
        self.patience = config["train"].as_int("patience")

        # optimizer
        self.optim = config["train"]["optim"]

        # if learning rate scheduling is required
        self.if_schedule = config["train"].as_bool("if_schedule")

        # N/A
        self.top_k = config["train"].as_int("top_k")

    def write_config(self, filename):
        """
        Write the details of the experiment in the form of a config file.
        This will be used to keep track of what experiments are running and 
        what parameters have been used.
        :return: 
        """
        # import json
        # with open(filename, 'w') as fp:
        #     json.dump(self.config.dict(), fp)
        self.config.filename = filename
        self.config.write()

    def get_all_attribute(self):
        """
        This function prints all the values of the attributes, just to cross
        check whether all the data types are correct.
        :return: Nothing, just printing
        """
        for attr, value in self.__dict__.items():
            print(attr, value)


if __name__ == "__main__":
    file = Config("config.yml")
    # file.write_config(sections, "config.yml")
    # print (file.config)
    print(file.write_config())