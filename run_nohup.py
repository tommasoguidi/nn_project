import os
import argparse
from pathlib import Path


def run_nohup(args):
    comb = {'fsl': {'method': ['proto', 'match', 'rel'], 'backbone': ['resnet', 'resnet18']},
            'instance_level': {'method': ['moe', 'naive'], 'backbone': []},
            'classifier': {'method': [], 'backbone': ['resnet', 'resnet18', 'cnn']}}

    assert args.script in comb, f'Gli script runnabili sono {[i for i in comb]}.'
    assert args.method in comb[args.script]['method'] if args.script != 'classifier' else True, \
        f'I method consentiti per {args.script} sono {comb[args.script]["method"]}.'
    assert args.backbone in comb[args.script]['backbone'] if args.script != 'instance_level' else True, \
        f'I method consentiti per {args.script} sono {comb[args.script]["backbone"]}.'
    dest = Path('runs') / args.script
    os.makedirs(dest, exist_ok=True)  # creo la directory se già non esiste
    past_experiments = len(os.listdir(dest))  # la prima si chiamerà run_0, la seconda run_1 e così via
    dest = dest / f'run_{past_experiments}'  # qui salvo i risultati degli esperimenti
    for i in range(args.n_folds):
        # makedirs crea tutte le cartelle intermedie che ancora non esistono specificate nel path
        # exists_ok fa si che se una cartella esiste già non c'è un errore
        os.makedirs(dest / f'fold_{i}', exist_ok=True)  # qui salvo i risultati del singolo split
    total_path = dest / 'run_events.out'

    command = f'nohup python {args.script}.py'
    for flag in vars(args):
        arg = getattr(args, flag)
        if flag != 'script' and arg != 'False':
            flag = flag.replace('_', '-')
            flag = '--' + flag
            command = ' '.join([command, flag, str(arg)])

    command += f' --nohup true --checkpoint-dir {str(dest)} > {total_path} &'

    print(command)
    os.system(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='(F) per fsl, (I) per instance_level, (C) per classifier',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--script', type=str, default='fsl',
                        help='Scegliere se runnare "fsl", "instance_level" o "classifier".')
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--root', type=str, default='/home/deepmammo/tommaso/prove/subset_10/',
                        help='Root del dataset. (C, I, F)')
    parser.add_argument('--n-folds', type=int, default=3, help='Numero di fold per la cross validation. (C, I, F)')
    parser.add_argument('-m', '--mode', type=str, default='train',
                        help='Scegliere tra "train" e "eval" in base a quale modalità si desidera. (C, I, F)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Scegliere se usare la gpu ("cuda") o la "cpu". (C, I, F)')
    parser.add_argument('-b', '--backbone', type=str, default='resnet',
                        help='Scegliere se utilizzare una semplice "cnn", "resnet" (50) o "resnet18" '
                             'come features extractor. (C, F)')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Epoche per eseguire il train. (C, I, F)')
    parser.add_argument('--batch-size', type=int, default=16, help='Numero di esempi in ogni batch. (C, I)')
    parser.add_argument('--num-workers', type=int, default=3, help='Numero di worker. (C, I, F)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate. (C, I, F)')
    parser.add_argument('--seed', type=int, default=123, help='Per riproducibilità. (C, I, F)')
    parser.add_argument('--weights', type=str, default='classifier.pth',
                        help='Percorso dei pesi da usare per il classificatore. (C, I, F)')
    parser.add_argument('--method', type=str, default='moe',
                        help='Scegliere se usare un approccio "naive" o "moe" se siamo in "instance_level", '
                             '"proto", "match" o "rel" se siamo in fsl. (I, F)')
    parser.add_argument('--pretrained', type=bool, default=False,
                        help='Per decidere se allenare anche la resnet o semplicemente caricare i pesi. (I)')
    parser.add_argument('--episodes', type=int, default=500, help='Numero di episodi per ogni epoca. (F)')
    parser.add_argument('--val-episodes', type=int, default=100,
                        help='Numero di episodi per ogni step di validazione. (F)')
    parser.add_argument('--test-episodes', type=int, default=1000, help='Numero di episodi per il test. (F)')
    parser.add_argument('--n-way', type=int, default=5, help='Numero di classi per ogni episodio. (F)')
    parser.add_argument('--k-shot', type=int, default=1, help='Numero di esempi per ogni classe nel support set. (F)')
    parser.add_argument('--n-query', type=int, default=4, help='Numero di esempi per ogni classe nel query set. (F)')

    arguments = parser.parse_args()
    run_nohup(arguments)
