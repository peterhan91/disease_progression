import argparse

def parser():
    parser = argparse.ArgumentParser(description='oai progressor prediction')


    parser.add_argument('--todo', choices=['train', 'valid', 'test', 'visualize'], default='train',
        help='what behavior want to do: train | valid | test | visualize')
    parser.add_argument('--dataset', default='oai', help='use what dataset')
    parser.add_argument('--subsample', type=float, default=0)
    parser.add_argument('--data_root', default='../OAI_Xray/dataset_most/imgs/', 
        help='the directory to save the dataset')
    parser.add_argument('--log_root', default='log', 
        help='the directory to save the logs or other imformations (e.g. images)')
    parser.add_argument('--model_root', default='checkpoint', help='the directory to save the models')
    parser.add_argument('--load_checkpoint', default='./oldmodel/')
    parser.add_argument('--affix', default='default', help='the affix for the save folder')


    parser.add_argument('--pretrain', type=bool, default=True, help='Use ImageNet pretraining or not')


    parser.add_argument('--batch_size', '-b', type=int, default=64, help='batch size')
    parser.add_argument('--max_epoch', '-m_e', type=int, default=300, 
        help='the maximum numbers of the model see a sample')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', '-w', type=float, default=0, 
        help='the parameter of l2 restriction for weights')


    parser.add_argument('--gpu', '-g', default='2', help='which gpu to use')
    parser.add_argument('--n_eval_step', type=int, default=10, 
        help='number of iteration per one evaluation')
    parser.add_argument('--n_checkpoint_step', type=int, default=5000, 
        help='number of iteration to save a checkpoint')
    parser.add_argument('--n_store_image_step', type=int, default=5000, 
        help='number of iteration to save adversaries')

    return parser.parse_args()


def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))