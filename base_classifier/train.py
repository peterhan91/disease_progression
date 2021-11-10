import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision as tv
from torchvision import models
from torch.autograd import Variable

from time import time
import dataset as patd
from utils import makedirs, create_logger, tensor2cuda, evaluate_, save_model
from argument import parser, print_args


class Trainer():
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

    def train(self, model, tr_loader, va_loader=None):
        args = self.args
        logger = self.logger
        opt = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[100, 200], gamma=0.1)
        _iter = 0
        begin_time = time()
        best_loss = 9999

        for epoch in range(1, args.max_epoch+1):
            for data, label in tr_loader:
                data, label = tensor2cuda(data), tensor2cuda(label)
                model.train()
                output = model(data)
                loss = F.binary_cross_entropy(torch.sigmoid(output), label)
                opt.zero_grad()
                loss.backward()
                opt.step() 

                if _iter % args.n_eval_step == 0:
                    logger.info('epoch: %d, iter: %d, spent %.2f s, tr_loss: %.3f' % (
                        epoch, _iter, time()-begin_time, loss.item()))
                    begin_time = time()

                if _iter % args.n_checkpoint_step == 0:
                    file_name = os.path.join(args.model_folder, 'checkpoint_%d.pth' % _iter)
                    save_model(model, file_name)

                _iter += 1
            scheduler.step()

            if va_loader is not None:
                t1 = time()
                va_acc, va_stdloss = self.test(model, va_loader, False)
                va_acc = va_acc * 100.0
                if va_stdloss < best_loss:
                    best_loss = va_stdloss
                    file_name = os.path.join(args.model_folder, 'checkpoint_best.pth')
                    save_model(model, file_name)
                t2 = time()
                logger.info('\n'+'='*20 +' evaluation at epoch: %d iteration: %d '%(epoch, _iter) + '='*20)
                logger.info('val acc: %.3f %%, spent: %.3f' % (va_acc, t2-t1))
                logger.info('val loss: %.3f, spent: %.3f' % (va_stdloss, t2-t1))
                logger.info('='*28+' end of evaluation '+'='*28+'\n')
    
    def test(self, model, loader, if_AUC=False):
        total_acc = 0.0
        total_stdloss = 0.0
        num = 0
        t = Variable(torch.Tensor([0.5]).cuda()) # threshold to compute accuracy
        label_list = []
        pred_list = []

        with torch.no_grad():
            for data, label in loader:
                data, label = tensor2cuda(data), tensor2cuda(label)
                model.eval()
                output = model(data)
                std_loss = F.binary_cross_entropy(torch.sigmoid(output), label)
                pred = torch.sigmoid(output)
                out = (pred > t).float()
                te_acc = np.mean(evaluate_(out.cpu().numpy(), label.cpu().numpy()))
                total_acc += te_acc
                total_stdloss += std_loss
                if if_AUC:
                    label_list.append(label.cpu().numpy())
                    pred_list.append(pred.cpu().numpy())
                num += 1
        if if_AUC:
            pred = np.squeeze(np.array(pred_list))
            label = np.squeeze(np.array(label_list))
            np.save(os.path.join(self.args.log_folder, 'y_pred.npy'), pred)
            np.save(os.path.join(self.args.log_folder, 'y_true.npy'), label)
        else:
            return total_acc / num, total_stdloss / num


def main(args):
    save_folder = '%s_%s' % (args.dataset, args.affix)
    log_folder = os.path.join(args.log_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)
    makedirs(log_folder)
    makedirs(model_folder)
    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)
    logger = create_logger(log_folder, args.todo, 'info')
    print_args(args, logger)

    model = models.resnet50(pretrained=args.pretrain)
    num_classes=1
    # for ResNet
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # for AlexNet and VGG
    # model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    if torch.cuda.is_available():
        model.cuda()

    trainer = Trainer(args, logger)
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    if args.todo == 'train':
        transform_train = tv.transforms.Compose([
                            tv.transforms.Resize(256),
                            tv.transforms.RandomHorizontalFlip(),
                            tv.transforms.ColorJitter(brightness=0.3, contrast=0.3, 
                                                        saturation=0.3, hue=0.3),
                            tv.transforms.RandomAffine(25, translate=(0.2, 0.2), 
                                                        scale=(0.8,1.2), shear=10),                            
                            tv.transforms.RandomCrop(224),
                            tv.transforms.ToTensor(),
                            tv.transforms.Normalize(means, stds)
                            ])

        transform_test = tv.transforms.Compose([
                            tv.transforms.Resize(256),
                            tv.transforms.CenterCrop(224),
                            tv.transforms.ToTensor(),
                            tv.transforms.Normalize(means, stds)
                            ])

        tr_dataset = patd.PatchDataset(path_to_images=args.data_root,
                                        fold='train', 
                                        sample=args.subsample,
                                        transform=transform_train)
        tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        va_dataset = patd.PatchDataset(path_to_images=args.data_root,
                                        fold='val',
                                        transform=transform_test)
        va_loader = DataLoader(va_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
             
        trainer.train(model, tr_loader, va_loader)
    
    elif args.todo == 'test':
        te_dataset = patd.PatchDataset(path_to_images=args.data_root,
                                        fold='test',
                                        transform=tv.transforms.Compose([
                                                    tv.transforms.Resize(256),
                                                    tv.transforms.CenterCrop(224),
                                                    tv.transforms.ToTensor(),
                                                    tv.transforms.Normalize(means, stds)
                                                    ]))
        te_loader = DataLoader(te_dataset, batch_size=1, shuffle=False, num_workers=1)
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint)
        std_acc, adv_acc = trainer.test(model, te_loader, if_AUC=True)
        print("std acc: %.4f, adv_acc: %.4f" % (std_acc * 100, adv_acc * 100))
    
    else:
        raise NotImplementedError
    

if __name__ == '__main__':
    args = parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
