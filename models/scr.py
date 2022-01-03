import torch
from utils.buffer_tricks_scr import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.loss_scr import SupConLoss
from backbone.ResNet18 import SupConResNet
import torchvision.transforms as transforms
from datasets import get_dataset
from torch.optim import SGD


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    parser.add_argument('--temper', type=float, default = 0.1, required=False,
                        help='Temperature.')
    #bs = 16, memory batch size = 64

    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class SCR(ContinualModel):
    NAME = 'scr'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(SCR, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.loss = SupConLoss(temperature=self.args.temper)
        self.net = SupConResNet()
        self.dataset = get_dataset(args)

        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
 
        self.class_means = None
        self.current_task = 0

    def observe(self, inputs, inputs_aug, labels, not_aug_inputs):
        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()

        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)

            inputs = torch.cat((inputs, buf_inputs))
            inputs_aug = torch.cat((inputs_aug, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        features = torch.cat([self.net.forward(inputs).unsqueeze(1), self.net.forward(inputs_aug).unsqueeze(1)], dim=1)

        loss = self.loss(features, labels)
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()

    def compute_class_means(self) -> None:
        """
        Computes a vector representing mean features for each class.
        """
        # This function caches class means
        transform = self.dataset.get_normalization_transform()
        class_means = []
        examples, labels = self.buffer.get_all_data(transform)
        for _y in self.classes_so_far:
            x_buf = torch.stack(
                [examples[i]
                 for i in range(0, len(examples))
                 if labels[i].cpu() == _y]
            ).to(self.device)

            class_means.append(self.net.features(x_buf).mean(0))
        self.class_means = torch.stack(class_means)

    def forward(self, x):
        if self.class_means is None:
            with torch.no_grad():
                self.compute_class_means()

        feats = self.net.features(x)
        feats = feats.unsqueeze(1)

        pred = (self.class_means.unsqueeze(0) - feats).pow(2).sum(2)
        return -pred

    def end_task(self, dataset) -> None:
        self.net.train()
        self.current_task += 1
        self.class_means = None
