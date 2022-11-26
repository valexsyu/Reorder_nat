import torch 
import subprocess
import sys


# name of the checkpoint.best_bleu_ file
num_updates = sys.argv[1]
task_name = sys.argv[2]

# num_updates = torch.load(file_name)['optimizer_history'][0]['num_updates']

result = subprocess.run(["tail",  "-1", f"tmp_score.{task_name}"], capture_output=True).stdout

bleu = float(result.split(b'\t')[2][:-1].strip())

print(f'{num_updates}\t{bleu}\n')

