import os, sys


task_name = sys.argv[1]
best_top5 = list()
for filename in os.listdir(f'checkpoints/{task_name}'):
    if filename.startswith('checkpoint.best_bleu_'):
        best_top5.append(float(filename.split('_')[2][:-3]))

if len(best_top5) == 5:
    least = sorted(best_top5)[0]
    print(f'checkpoint.best_bleu_{least}.pt')
else:
    print('None')
