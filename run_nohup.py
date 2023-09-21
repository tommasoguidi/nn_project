import os
import argparse
from pathlib import Path


def run_nohup(args):
    script = Path(args.script)
    dest = Path('runs') / script.stem
    os.makedirs(dest, exist_ok=True)  # creo la directory se già non esiste
    past_experiments = len(os.listdir(dest))  # la prima si chiamerà run_0, la seconda run_1 e così via
    dest = dest / f'run_{past_experiments}'  # qui salvo i risultati degli esperimenti
    for i in range(args.n_folds):
        # makedirs crea tutte le cartelle intermedie che ancora non esistono specificate nel path
        # exists_ok fa si che se una cartella esiste già non c'è un errore
        os.makedirs(dest / f'fold_{i}', exist_ok=True)  # qui salvo i risultati del singolo split
    total_path = dest / 'run_events.out'

    assert args.modality in ['train', 'evaluate'], ''
    command = f'nohup python {args.modality}.py'
    for flag in vars(args):

        if flag != 'modality':
            arg = getattr(args, flag)
            flag = flag.replace('_', '-')
            flag = '--' + flag
            command = ' '.join([command, flag, str(arg)])

    command += f' > {total_path} &'

    print(command)
    os.system(command)



parser = argparse.ArgumentParser(description='Arguments to be passed to the train.py script, in the help (B) means that the argument is used by both scripts, while (T) and (E) specify specific arguments.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--device', default='cuda', help='device to use for prediction. (B)')
parser.add_argument('--id-gpu', default='0', help='id of the gpu to use for prediction. (B)')
parser.add_argument('--seed', type=int, default=87, help='seed for reproducibility. (B)')
parser.add_argument('--root', type=str, default='/home/deepmammo/Data/DEEPMAMMOSET/CDD_CESM', help='root of the dataset. (B)')
# --split non è usato in train
parser.add_argument('--split', type=str, default='test', help='choose if evaluate on "test" or "validation". (E)')
parser.add_argument('--cross-val-num-buckets', type=int, default=5, help='num of buckets for cross validation. (B)')
parser.add_argument('--cross-val-bucket-index', type=int, default=0, help='index of bucket for validation; remaining are used for training. (B)')
parser.add_argument('--image-size', type=int, default=224, help='dimension to which image will be resized. (B)')
parser.add_argument('--batch-size', type=int, default=64, help='batch size. (B)')
parser.add_argument('--num-workers', type=int, default=16, help='num workers. (B)')
parser.add_argument('--model', type=str, default='resnet', help='model to be used, either "resnet", "vgg" or "efficientnet_v2". (B)')
parser.add_argument('--arch', type=str, default='resnet101', help='architecture to be used, either "resnet50", "resnet101", "resnet152", "vgg13", "vgg16", "vgg19", "efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l". (B)')
# --pretrained e --num-epochs non sono usati in evaluate
parser.add_argument('--pretrained', default=True, help='load pretrained model (on ImageNet). (T)')
parser.add_argument('--num-epochs', type=int, default=50, help='num epochs. (T)')
# --weights e --aggregation non sono usati in train
parser.add_argument('--weights', type=str, default='last.pth', help='path to the model. (E)')
parser.add_argument('--aggregation', default='max', help='method used for aggregating the results of multiple views in a single result per side (one of "mean", "min", "max"). (E)')
# --val-freq non è usato in evaluate
parser.add_argument('--val-freq', type=int, default=1, help='validation frequency. (T)')
parser.add_argument('--val-method', default='accuracy', help='evaluation method, either "accuracy" or "pfscore". (B)')
# --batch-accumulation e --log-every non sono usati in evaluate
parser.add_argument('--batch-accumulation', type=int, default=2, help='num of iteration for batch accumulation. (T)')
parser.add_argument('--log-every', type=int, default=2, help='num of iteration after which logging. (T)')
parser.add_argument('--debug', default=True, help='enable debugging in validation. (B)')
parser.add_argument('--debug-every', type=int, default=2, help='num of iteration after which debugging during val. (B)')
parser.add_argument('--pos-weight', type=int, default=1, help='weight to be associated to positive examples. (B)')
# --annotated non è usato in train
parser.add_argument('--modality', type=str, default='train', help='choose to run train.py or evaluate.py. (Only for nohup)')

args = parser.parse_args()
run_nohup(args)
