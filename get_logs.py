from pathlib import Path
import shutil
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt


src_dir = Path('runs')
dest_dir = Path('nice_logs')
interessanti = {'classifier': [7, 23],
                'instance_level': [3, 5, 9, 10, 28, 29, 31, 34, 36, 37],
                'fsl': [5, 20, 25, 27, 38, 46]}

for script in interessanti:
    for run in interessanti[script]:
        for i in range(3):
            src_path = src_dir / script / f'run_{str(run)}' / f'fold_{str(i)}'
            file = str(src_path.glob('*deepmammo*').__next__())
            dest_path = dest_dir / script / f'run_{str(run)}' / f'fold_{str(i)}'
            os.makedirs(dest_path, exist_ok=True)
            dest = dest_path / file.split('/')[-1]
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

            for i in data:
                epochs = data[i][0]
                values = data[i][1]
                pic_name = i.replace('/', '_')

                plt.plot(epochs, values)
                plt.suptitle(f'{i}')
                plt.savefig(dest_path / f'{pic_name}.png')
