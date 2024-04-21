from __future__ import absolute_import, division, print_function
import logging
import argparse
import os
import random
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models.model import LocalViT, AdversarialNetwork
from tool.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from tool.transform import get_transform
from tool.utils import visda_acc
from data.data_list_image import ImageList, ImageListIndex
from models.Loss import *
import models.configs as configs
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


def save_model(args, model, is_adv=False):
    model_to_save = model.module if hasattr(model, 'module') else model
    if not is_adv:
        model_checkpoint = os.path.join(args.output_dir, args.dataset, "%s_checkpoint.bin" % args.name)
    else:
        model_checkpoint = os.path.join(args.output_dir, args.dataset, "%s_checkpoint_adv.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", os.path.join(args.output_dir, args.dataset))


def setup(args):
    # Prepare model
    config = configs.get_b16_config()
    model = LocalViT(config, args.img_size, zero_head=True, 
                              num_classes=args.num_classes, msa_layer=args.msa_layer)
    # model.load_state_dict(torch.load("checkpoint/localvit.pth"),strict=False)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
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


def valid(args, model, ad_net, writer, test_loader, global_step):
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    ad_net.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            _, logits, _, _ = model(x, ad_net=ad_net)

            eval_loss = loss_fct(logits, y)
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
    if args.dataset == 'SAR3class': 
        accuracy, classWise_acc = visda_acc(all_preds, all_label)
    else :
        accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results of: %s" % args.name)
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    if args.dataset == 'SAR3class' :
        return accuracy, classWise_acc
    else :
        return accuracy, None
def train(args, model):
    if args.local_rank in [-1, 0]:
        os.makedirs(os.path.join(args.output_dir, args.dataset), exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.dataset, args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    transform_source, transform_target, transform_test = get_transform(args.dataset, args.img_size)
    source_loader = torch.utils.data.DataLoader(
        ImageList(open(args.source_list).readlines(), transform=transform_source, mode='RGB'),
        batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    
    target_loader = torch.utils.data.DataLoader(
        ImageListIndex(open(args.target_list).readlines(), transform=transform_target, mode='RGB'),
        batch_size=args.train_batch_size, shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(
        ImageList(open(args.test_list).readlines(), transform=transform_test, mode='RGB'),
        batch_size=args.eval_batch_size, shuffle=False, num_workers=4)
    
    config = configs.get_b16_config()
    ad_net = AdversarialNetwork(config.hidden_size, config.hidden_size//4)
    ad_net.to(args.device)
    ad_net_local = AdversarialNetwork(config.hidden_size//12, config.hidden_size//12)
    ad_net_local.to(args.device)
    
    optimizer_ad = torch.optim.SGD(list(ad_net.parameters())+list(ad_net_local.parameters()),
                            lr=args.learning_rate/10, 
                            momentum=0.9,
                            weight_decay=args.weight_decay)
    
    optimizer = torch.optim.SGD([
                                    {'params': model.transformer.parameters(), 'lr': args.learning_rate/10},
                                    {'params': model.head.parameters()},
                                ],
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        scheduler_ad = WarmupCosineSchedule(optimizer_ad, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        scheduler_ad = WarmupLinearSchedule(optimizer_ad, warmup_steps=args.warmup_steps, t_total=t_total)
        
    model.zero_grad()
    ad_net.zero_grad()
    ad_net_local.zero_grad()
        
    set_seed(args)  
    best_acc = 0
    best_classWise_acc = ''
    accuracy = 0
    len_source = len(source_loader)
    len_target = len(target_loader)            
    cls_criterion = RSCM_Loss(args.num_classes).cuda()

    
    for global_step in range(1, t_total):
        model.train()
        ad_net.train()
        ad_net_local.train()
        Lambda = 0.25 * global_step/t_total 
        cfam_loss = CFAMLoss(gamma=0.5)
        if (global_step-1) % (len_source-1) == 0:
            iter_source = iter(source_loader)    
        if (global_step-1) % (len_target-1) == 0:
            iter_target = iter(target_loader)
        
        data_source = iter_source.next()
        data_target = iter_target.next()

        x_s, y_s = tuple(t.to(args.device) for t in data_source)
        x_t, _, index_t = tuple(t.to(args.device) for t in data_target)
        feature_s, feature_t, logits_s, logits_t, loss_ad_local, x_s, x_t = model(x_s, x_t, ad_net_local)
        
        pseudo_labels = logits_t.argmax(dim=1)
        if args.use_RSCM:#RSCM
            memory_source_features = feature_s
            memory_source_labels = y_s
            memory_target_features = feature_t
            memory_target_labels = pseudo_labels
            class_num = args.num_classes  #3

            mean_source = CalculateMean(memory_source_features, memory_source_labels, class_num)
            mean_target = CalculateMean(memory_target_features, memory_target_labels, class_num)
            cv_target = Calculate_CV(memory_target_features, memory_target_labels, mean_target, class_num)
            loss_clc = cls_criterion(model.head, feature_s, logits_s, y_s, Lambda, mean_source, mean_target, cv_target)
            loss_fct = CrossEntropyLoss()
            loss_clc = loss_fct(logits_s.view(-1, args.num_classes), y_s.view(-1)) + 0.1 * loss_clc
        else:
            loss_fct = CrossEntropyLoss()
            loss_clc = loss_fct(logits_s.view(-1, args.num_classes), y_s.view(-1))

        targets_concat = torch.cat((y_s.view(-1), pseudo_labels), dim=0)

        features_mix = torch.cat((feature_s, feature_t), dim=0)
        if args.use_CFAM:#CFAM
            loss_cfam = cfam_loss(features_mix, targets_concat)  

        loss_ad_global = adv(torch.cat((x_s[:,0], x_t[:,0]), 0), ad_net)
        loss = loss_clc + args.beta * loss_ad_global + args.gamma * loss_ad_local
        if args.use_CFAM:
            loss += 0.01 * loss_cfam  
            
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(ad_net.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(ad_net_local.parameters(), args.max_grad_norm)
        
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        
        optimizer_ad.step()
        optimizer_ad.zero_grad()
        scheduler_ad.step()
        
        if args.local_rank in [-1, 0]:
            writer.add_scalar("train/loss", scalar_value=loss.item(), global_step=global_step)
            writer.add_scalar("train/loss_clc", scalar_value=loss_clc.item(), global_step=global_step)
            writer.add_scalar("train/loss_ad_global", scalar_value=loss_ad_global.item(), global_step=global_step)
            writer.add_scalar("train/loss_ad_local", scalar_value=loss_ad_local.item(), global_step=global_step)
            writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=global_step)
        
        if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
            accuracy, classWise_acc = valid(args, model, ad_net_local, writer, test_loader, global_step)
            if best_acc < accuracy:
                save_model(args, model)
                save_model(args, ad_net_local, is_adv=True)
                best_acc = accuracy

                if classWise_acc is not None:
                    best_classWise_acc = classWise_acc
            model.train()
            ad_net_local.train()
            logger.info("Current Best Accuracy: %2.5f" % best_acc)
            logger.info("Current Best element-wise acc: %s" % best_classWise_acc)
        
    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("Best element-wise Accuracy: \t%s" % best_classWise_acc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", help="Which downstream task.")
    parser.add_argument("--source_list", help="Path of the training data.")
    parser.add_argument("--target_list", help="Path of the test data.")
    parser.add_argument("--test_list", help="Path of the test data.")
    parser.add_argument("--num_classes", default=3, type=int,
                        help="Number of classes in the dataset.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=256, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=6, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    parser.add_argument("--beta", default=0.1, type=float,
                        help="The importance of the adversarial loss.")
    parser.add_argument("--gamma", default=0., type=float,
                        help="The importance of the local adversarial loss.")
    parser.add_argument("--msa_layer", default=12, type=int,
                        help="The layer that incorporates local alignment.")
    parser.add_argument("--use_CFAM", default=False, action="store_true",
                        help="Use CFAM loss.")
    parser.add_argument("--use_RSCM", default=False, action="store_true",
                        help="Use RSCM loss.")
    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=5000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="linear",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    # Setup CUDA, GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))
    
    set_seed(args)
    args, model = setup(args)
    model.to(args.device)
    train(args, model)


if __name__ == "__main__":
    main()
