from pathlib import Path
import shutil
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
from tqdm import tqdm


src_dir = Path('runs')
dest_dir = Path('nice_logs')
interessanti = {'classifier': [2, 3, 5, 7, 23, 27],
                'instance_level': [3, 5, 9, 10, 28, 29, 31, 34, 36, 37],
                'fsl': [5, 20, 25, 27, 38, 46, 50, 51, 52, 53, 54, 56, 57]}

for script in tqdm(interessanti, total=3, leave=False, desc='SALVANDO LE LOG PIU INTERESSANTI...'):
    for run in tqdm(interessanti[script], leave=False, desc=f'...TRA QUELLE DI {script}'):
        for i in range(3):
            src_path = src_dir / script / f'run_{str(run)}' / f'fold_{str(i)}'
            file = str(src_path.glob('*deepmammo*').__next__())
            dest_path = dest_dir / script / f'run_{str(run)}' / f'fold_{str(i)}'
            os.makedirs(dest_path, exist_ok=True)
            dest = dest_path / file.split('/')[-1]
            if not os.path.exists(dest):
                shutil.copy(file, dest)

                event_acc = EventAccumulator(file)
                event_acc.Reload()
                data = {}

                hparam_file = False  # I save hparam files as 'hparam/xyz_metric'
                for tag in sorted(event_acc.Tags()["scalars"]):
                    if tag.split('/')[0] == 'hparam': hparam_file = True  # check if its a hparam file
                    step, value = [], []

                    for scalar_event in event_acc.Scalars(tag):
                        step.append(scalar_event.step)
                        value.append(scalar_event.value)

                    data[tag] = (step, value)

                for j in data:
                    epochs = data[j][0]
                    values = data[j][1]
                    pic_name = j.replace('/', '_')

                    fig = plt.figure()
                    plt.plot(epochs, values)
                    plt.suptitle(f'{j}')
                    fig.savefig(dest_path / f'{pic_name}.png')
                    plt.close(fig)
