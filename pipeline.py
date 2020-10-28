from references.detection.engine import train_one_epoch, evaluate

class PipelineFasterRCNN():
    def __init__(self, num_epochs, model, optimizer,
                data_loader, data_loader_test, device, print_freq=1):

        self.num_epochs = num_epochs
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.data_loader_test = data_loader_test
        self.device = device
        self.print_freq = print_freq

    def train(self):
        for epoch in range(self.num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(self.model,
                            self.optimizer,
                            self.data_loader,
                            self.device,
                            epoch,
                            self.print_freq)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(self.model,
                     self.data_loader_test,
                     device=device)


class PipelineSSD300():
    def __init__(self, num_epochs, model, optimizer,
                data_loader, data_loader_test, device, print_freq=1):

        self.num_epochs = num_epochs
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.data_loader_test = data_loader_test
        self.device = device
        self.print_freq = print_freq

    def train(self):
        for epoch in range(self.num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(self.model,
                            self.optimizer,
                            self.data_loader,
                            self.device,
                            epoch,
                            self.print_freq)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(self.model,
                     self.data_loader_test,
                     device=device)
