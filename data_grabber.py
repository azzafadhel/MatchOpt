from util import *
from glob import glob

from pprint import pprint

CACHE_FOLDER = '/home/afadhel_l/home1/Design-baselines/Design_baselines/'

METHOD_NAMES = {
                'autofocused-cbas',
                'bo-qei',
                'cma-es',
                'reinforce',
                'cbas',
                'gradient-ascent',
                'gradient-ascent-mean-ensemble',
                'gradient-ascent-min-ensemble',
                'mins',
                'coms'

               }

TASK_NAMES = {
              'ant',
              'dkitty',
              'hopper',
              'superconductor',
              'tf-bind-8',
              'tf-bind-10',
              #'gfp',
              #'utr',

             }

# folders = glob(CACHE_FOLDER + '*/', recursive = True)
# pprint(folders)

data = dict()

for algo in METHOD_NAMES:
    data[algo] = dict()
    for task in TASK_NAMES:
        print ('TASK_NAMES', TASK_NAMES)
        # print(CACHE_FOLDER + algo + '-' + task + '/*/')
        # print(glob(CACHE_FOLDER + algo + '-' + task + '/*/'))
        try:
            trials = glob(glob(CACHE_FOLDER + algo + '-' + task + '/*/')[0] + '*/')
            #trials=glob(CACHE_FOLDER + algo + '-' + task + '/*/')


            print ('##############################')
            #print(trials)
            #print ('1')
            algo_task = dict()
            for i, trial in enumerate(trials):
                print(trial)
                trial_data = load_object(trial + 'data/hacked.dat')
                algo_task[i] = trial_data
                print("##########")
            data[algo][task] = algo_task
        except Exception as e:
            print(CACHE_FOLDER + algo + '-' + task + '/*/')


save_object(data, CACHE_FOLDER + 'summarized_data.dat')


