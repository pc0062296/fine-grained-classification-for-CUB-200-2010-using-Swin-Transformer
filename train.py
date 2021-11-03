#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import logging
import argparse
import os
import random
import numpy as np
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.utils.data import random_split, RandomSampler, SequentialSampler
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
import timm
from PIL import Image

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    rt /= nprocs
    return rt


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir,
                                    "%s_checkpoint.bin" % args.name)
    if args.fp16:
        checkpoint = {
            'model': model_to_save.state_dict(),
            'amp': amp.state_dict()
        }
    else:
        checkpoint = {
            'model': model_to_save.state_dict(),
        }

    torch.save(checkpoint, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    # Prepare model
    num_classes = 200

    model = timm.models.swin_large_patch4_window12_384_in22k(pretrained=True)
    # a swin transformer model from timm

    model.head = torch.nn.Linear(in_features=1536, out_features=200, bias=True)
    model.to(args.device)
    num_params = count_parameters(model)
    print(model)
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class birdDataset(Dataset):
    def __init__(self, labels, imgs, transform):
        self.transform = transform
        self.lbls = labels
        self.imgs = imgs
        self.imgs = [f'dataset/train/{i}' for i in self.imgs]
        assert len(self.imgs) == len(self.lbls), 'mismatched length!'
        print('Total data in {}'.format(len(self.imgs)))

    def __getitem__(self, index):
        imgpath = self.imgs[index]
        imgg = Image.open(imgpath).convert('RGB')
        lbl = int(self.lbls[index])-1
        imgg = self.transform(imgg)

        return imgg, lbl

    def __len__(self):
        return len(self.imgs)


def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=False,
                          position=0,
                          leave=True)
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)

            eval_loss = loss_fct(logits, y)
            eval_loss = eval_loss.mean()
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)
    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    accuracy = torch.tensor(accuracy).to(args.device)

    val_accuracy = reduce_mean(accuracy, args.nprocs)
    val_accuracy = val_accuracy.detach().cpu().numpy()

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % val_accuracy)

    return val_accuracy


def train(args, model, train_loader, test_loader):
    """ Train the model """

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20
        args.fp16_isact = True

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size = %d",
                args.train_batch_size)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=False)
        all_preds, all_label = [], []
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch

            # loss, logits = model(x, y)
            logits = model(x)
            loss = torch.nn.CrossEntropyLoss()(logits, y)
            loss = loss.mean()
            preds = torch.argmax(logits, dim=-1)

            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(y.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(
                    all_preds[0], preds.detach().cpu().numpy(), axis=0
                )
                all_label[0] = np.append(
                    all_label[0], y.detach().cpu().numpy(), axis=0
                )

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())

            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
            )

            writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
            writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
            if global_step % args.eval_every == 0:
                with torch.no_grad():
                    accuracy = valid(args, model, writer, test_loader, global_step)

                if best_acc < accuracy:
                    save_model(args, model)
                    best_acc = accuracy
                logger.info("best accuracy so far: %f" % best_acc)
                model.train()

            if global_step % t_total == 0:
                break

        all_preds, all_label = all_preds[0], all_label[0]
        accuracy = simple_accuracy(all_preds, all_label)
        accuracy = torch.tensor(accuracy).to(args.device)
        train_accuracy = reduce_mean(accuracy, args.nprocs)
        train_accuracy = train_accuracy.detach().cpu().numpy()
        logger.info("train accuracy so far: %f" % train_accuracy)

        losses.reset()
        if global_step % t_total == 0:
            break

    writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")

# In[3]:

parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument("--name", required=False, default="hw1",
                    help="Name of this run. Used for monitoring.")
parser.add_argument("--output_dir", default="output/", type=str,
                    help="The output directory where checkpoints will be written.")
parser.add_argument("--train_batch_size", default=16, type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size", default=8, type=int,
                    help="Total batch size for eval.")
parser.add_argument("--eval_every", default=100, type=int,
                    help="Run prediction on validation set every so many steps."
                         "Will always run one evaluation at the end of training.")
parser.add_argument("--learning_rate", default=3e-2, type=float,
                    help="The initial learning rate for SGD.")
parser.add_argument("--weight_decay", default=0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--num_steps", default=40000, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                    help="How to decay the learning rate.")
parser.add_argument("--warmup_steps", default=100, type=int,
                    help="Step of training to perform learning rate warmup for.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")
parser.add_argument('--fp16', action='store_true', default=True,
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--fp16_opt_level', type=str, default='O2',
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument('--loss_scale', type=float, default=0,
                    help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                         "0 (default value): dynamic loss scaling.\n"
                         "Positive power of 2: static loss scaling value.\n")
parser.add_argument('--smoothing_value', type=float, default=0.0,
                    help="Label smoothing value\n")

args, unknown = parser.parse_known_args()

# Setup CUDA, GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.n_gpu = torch.cuda.device_count()

args.device = device
args.nprocs = torch.cuda.device_count()

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO if True else logging.WARN)
logger.warning("Process  device: %s, n_gpu: %s" % (args.device, args.n_gpu))

# In[4]:

# Set seed
set_seed(args)

# Model & Tokenizer Setup
args, model = setup(args)

# In[5]:

# Prepare dataset
imgids = []
labels = []
with open('dataset/training_labels.txt', "r", encoding="utf-8") as f:
    for line in f.readlines():
        strr = line.split(" ")
        imgids.append(strr[0])
        labels.append(strr[1][0:3])

# evenly distribute to training and testing 4:1
training_imgids = []
training_labels = []
testing_imgids = []
testing_labels = []
number = np.zeros(200, dtype=int)
for i in range(3000):
    if number[int(labels[i])-1] < 12:
        training_labels.append(labels[i])
        training_imgids.append(imgids[i])
        number[int(labels[i])-1] += 1
    else:
        testing_labels.append(labels[i])
        testing_imgids.append(imgids[i])
        number[int(labels[i])-1] += 1

train_transform = transforms.Compose([transforms.Resize((420, 420), Image.BILINEAR),
                                     # transforms.RandomResizedCrop(384),
                                     transforms.RandomCrop((384, 384)),
                                     # transforms.CenterCrop((384, 384)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(10),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_transform = transforms.Compose([transforms.Resize((420, 420), Image.BILINEAR),
                                    transforms.CenterCrop((384, 384)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

trainset = birdDataset(training_labels, training_imgids, transform=train_transform)
testset = birdDataset(testing_labels, testing_imgids, transform=test_transform)
train_sampler = RandomSampler(trainset)
test_sampler = SequentialSampler(testset)

train_loader = DataLoader(trainset,
                          sampler=train_sampler,
                          batch_size=args.train_batch_size,
                          # num_workers=0,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(testset,
                         sampler=test_sampler,
                         batch_size=args.eval_batch_size,
                         # num_workers=0,
                         pin_memory=True) if testset is not None else None

# In[6]:

# pertrained model
pretrained_model = torch.load("output/hw1model/hw1model.bin")['model']
model.load_state_dict(pretrained_model)

# In[7]:

# freeze Training

for p in model.parameters():
    p.requires_grad = False

for p in model.head.parameters():
    p.requires_grad = True

for p in model.norm.parameters():
    p.requires_grad = True

for p, module in model.layers.named_children():
    if p == '3':
        for r in module.parameters():
            r.requires_grad = True
    if p == '2':
        for r in module.parameters():
            r.requires_grad = False

args.warmup_steps = 50
args.train_batch_size = 2
args.eval_batch_size = 2
args.learning_rate = 0.01
args.num_steps = 10000
args.eval_every = int(2400/args.train_batch_size)
args.weight_decay = 0.0
args.name = "freeze"

train(args, model, train_loader, test_loader)
# end