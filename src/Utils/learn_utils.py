"""
Defines learning rate schedulers.
"""

class LearningRate:
    """
    utils functions to manipulate the learning rate
    """

    def __init__(self, optimizer, init_lr=0.01, lr_decay_epoch=10,
                 lr_dacay_fact=0.1,
                 patience=10,
                 logger=None):
        """
        :param optimizer: Object of the torch optimizer initialized before
        :param init_lr: Start lr
        :param lr_decay_epoch: Epchs after which the learning rate to be decayed
        :param lr_dacay_fact: Factor by which lr to be decayed
        :param patience: Number of epochs to wait for the loss to decrease 
        before reducing the lr

        """
        self.opt = optimizer
        self.init_lr = init_lr
        self.lr_dacay_fact = lr_dacay_fact
        self.lr_decay_ep = lr_decay_epoch
        self.loss = 1000
        self.patience = patience
        self.pat_count = 0
        self.lr = init_lr
        self.logger = logger
        pass

    def exp_lr_scheduler(self, epoch):
        """Decay learning rate by a factor of 0.1 every lr_decay_epoch 
        epochs. This is done irrespective of the loss.
        :param epoch: Current epoch number
        :return: """
        if epoch % self.lr_decay_ep == 0 and epoch > 0:
            self.red_lr_by_fact()
        return self.opt

    def red_lr_by_fact(self):
        """
        reduces the learning rate by the pre-specified factor
        :return: 
        """
        self.lr = self.lr * self.lr_dacay_fact
        for param_group in self.opt.param_groups:
            param_group['lr'] = self.lr
        if self.logger:
            self.logger.info('LR is set to {}'.format(self.lr))
        else:
            print('LR is set to {}'.format(self.lr))

    def reduce_on_plateu(self, loss):
        """
        Reduce the learning rate when loss doesn't decrease
        :param loss: loss to be monitored
        :return: optimizer with new lr
        """
        if self.loss > loss:
            self.loss = loss
            self.pat_count = 0
        else:
            self.pat_count += 1
            if self.pat_count > self.patience:
                self.pat_count = 0
                self.red_lr_by_fact()
