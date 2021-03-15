import torch

# FasterRCNN Imports
from references.detection.engine import train_one_epoch_FRCNN, evaluate

# SSD Imports
from references.detection.engine import train_one_epoch_SSD, evaluate
from SSD.SSD_nvidia.src.utils import dboxes300_coco, Encoder
from SSD.SSD_nvidia.src.train import tencent_trick
from SSD.SSD_nvidia.src.model import Loss

class PipelineFasterRCNN():
    def __init__(self,
                num_epochs,
                model,
                lr,
                momentum,
                weight_decay,
                data_loader,
                data_loader_test,
                device,
                print_freq=1):

        self.num_epochs = num_epochs
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.data_loader = data_loader
        self.data_loader_test = data_loader_test
        self.device = device
        self.print_freq = print_freq

    def train(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        for epoch in range(self.num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch_FRCNN(self.model,
                            optimizer,
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
    def __init__(self,
                num_epochs,
                model,
                lr,
                momentum,
                weight_decay,
                data_loader,
                data_loader_test,
                device,
                print_freq=1):

        self.num_epochs = num_epochs
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.data_loader = data_loader
        self.data_loader_test = data_loader_test
        self.device = device
        self.print_freq = print_freq

    def train(self):
        dboxes = dboxes300_coco()
        encoder = Encoder(dboxes)

        loss_func = Loss(dboxes)

        # Don't think this is needed since we do model.to(device) earlier
        # if self.device == "cuda":
        #     self.model.cuda()
        #     loss_func.cuda()


        optimizer = torch.optim.SGD(tencent_trick(self.model), lr=self.lr,
                                                momentum=self.momentum,
                                                weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        # Try ...to(device) here too?
        # if self.device == "cuda":
        #     from apex import amp
        #     self.model, optimizer = amp.initialize(self.model, optimizer, opt_level='O2')

        mean, std = self.generate_mean_std()

        for epoch in range(self.num_epochs):
            train_one_epoch_SSD(self.model,
                                loss_func,
                                optimizer,
                                self.data_loader,
                                encoder,
                                epoch,
                                self.print_freq,
                                mean,
                                std,
                                self.device)

            # lr_scheduler.step()



        #     end_epoch_time = time.time() - start_epoch_time
        #     total_time += end_epoch_time
        #
        #     if args.local_rank == 0:
        #         logger.update_epoch_time(epoch, end_epoch_time)
        #
        #     if epoch in args.evaluation:
        #         acc = evaluate(ssd300, val_dataloader, cocoGt, encoder, inv_map, args)
        #
        #         if args.local_rank == 0:
        #             logger.update_epoch(epoch, acc)
        #
        #     if args.save and args.local_rank == 0:
        #         print("saving model...")
        #         obj = {'epoch': epoch + 1,
        #                'iteration': iteration,
        #                'optimizer': optimizer.state_dict(),
        #                'scheduler': scheduler.state_dict(),
        #                'label_map': val_dataset.label_info}
        #         if args.distributed:
        #             obj['model'] = ssd300.module.state_dict()
        #         else:
        #             obj['model'] = ssd300.state_dict()
        #         save_path = os.path.join(args.save, f'epoch_{epoch}.pt')
        #         torch.save(obj, save_path)
        #         logger.log('model path', save_path)
        #     train_loader.reset()
        # DLLogger.log((), { 'total time': total_time })
        # logger.log_summary()






    def generate_mean_std(self):
        mean_val = [0.485, 0.456, 0.406]
        std_val = [0.229, 0.224, 0.225]

        # mean = torch.tensor(mean_val).cuda()
        # std = torch.tensor(std_val).cuda()

        mean = torch.tensor(mean_val)
        std = torch.tensor(std_val)

        view = [1, len(mean_val), 1, 1]

        mean = mean.view(*view)
        std = std.view(*view)

        # if args.amp:
        #     mean = mean.half()
        #     std = std.half()

        return mean, std
