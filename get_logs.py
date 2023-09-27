from pathlib import Path
import shutil
import os


src_dir = Path('runs')
dest_dir = Path('nice_logs')
interessanti = {'classifier': [7, 23],
		'instance_level': [3, 5, 9, 10, 28, 29, 31, 34, 36],
		'fsl': [5, 20, 25, 27, 38]}

for script in interessanti:
	for run in interessanti[script]:
		for i in range(3):
			src_path = src_dir / script / f'run_{str(run)}' / f'fold_{str(i)}'
			files = [str(i) for i in src_path.glob('*')]
			for file in files:
				if 'deepmammo' in file:
					dest_path = dest_dir / script / f'run_{str(run)}' / f'fold_{str(i)}'
					os.makedirs(dest_path, exist_ok=True)
					dest = dest_path / file.split('/')[-1]
					shutil.copy(file, dest)
