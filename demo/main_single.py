import os
import sys
import time
import argparse
import multiprocessing
from utils import Path_utils
from utils import os_utils
from pathlib import Path

if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 6):
    raise Exception("This program requires at least Python 3.6")

class fixPathAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

#if __name__ == "__main__":
#multiprocessing.set_start_method("spawn")

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()




#def process_train(arguments):
#    os_utils.set_process_lowest_prio()
#    args = {'training_data_src_dir'  : arguments.training_data_src_dir,
#            'training_data_dst_dir'  : arguments.training_data_dst_dir,
#            'pretraining_data_dir'   : arguments.pretraining_data_dir,
#            'model_path'             : arguments.model_dir,
#            'model_name'             : arguments.model_name,
#            'no_preview'             : arguments.no_preview,
#            'debug'                  : arguments.debug,
#            'execute_programs'       : [ [int(x[0]), x[1] ] for x in arguments.execute_program ]
#            }
#    device_args = {'cpu_only'  : arguments.cpu_only,
#                   'force_gpu_idx' : arguments.force_gpu_idx,
#                   }
#    from mainscripts import Trainer
#    Trainer.main(args, device_args)



def process_train(args,devive_args):
    os_utils.set_process_lowest_prio()
#    args = {'training_data_src_dir'  : arguments.training_data_src_dir,
#            'training_data_dst_dir'  : arguments.training_data_dst_dir,
#            'pretraining_data_dir'   : arguments.pretraining_data_dir,
#            'model_path'             : arguments.model_dir,
#            'model_name'             : arguments.model_name,
#            'no_preview'             : arguments.no_preview,
#            'debug'                  : arguments.debug,
#            'execute_programs'       : [ [int(x[0]), x[1] ] for x in arguments.execute_program ]
#            }
#    device_args = {'cpu_only'  : arguments.cpu_only,
#                   'force_gpu_idx' : arguments.force_gpu_idx,
#   
    args={'training_data_src_dir': '/home/heimi/aaDeepFaceLab/workspace/data2_src/aligned',
     'training_data_dst_dir': '/home/heimi/aaDeepFaceLab/workspace/data2_dst/aligned', 
     'pretraining_data_dir': None, 
     'model_path': '/home/heimi/aaDeepFaceLab/workspace/model2',
     'model_name': 'H64', 'no_preview': False, 
     'debug': False, 
     'execute_programs': []}
    
    
    device_args={'cpu_only': True,
                 'force_gpu_idx': -1}
               
    from mainscripts import Trainer
    Trainer.main(args, device_args)
    
    
    

p = subparsers.add_parser( "train", help="Trainer")
p.add_argument('--training-data-src-dir', required=True, action=fixPathAction, dest="training_data_src_dir", help="Dir of extracted SRC faceset.")
p.add_argument('--training-data-dst-dir', required=True, action=fixPathAction, dest="training_data_dst_dir", help="Dir of extracted DST faceset.")
p.add_argument('--pretraining-data-dir', action=fixPathAction, dest="pretraining_data_dir", default=None, help="Optional dir of extracted faceset that will be used in pretraining mode.")
p.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir", help="Model dir.")
p.add_argument('--model', required=True, dest="model_name", choices=Path_utils.get_all_dir_names_startswith ( Path(__file__).parent / 'models' , 'Model_'), help="Type of model")
p.add_argument('--no-preview', action="store_true", dest="no_preview", default=False, help="Disable preview window.")
p.add_argument('--debug', action="store_true", dest="debug", default=False, help="Debug samples.")
p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Train on CPU.")
p.add_argument('--force-gpu-idx', type=int, dest="force_gpu_idx", default=-1, help="Force to choose this GPU idx.")
p.add_argument('--execute-program', dest="execute_program", default=[], action='append', nargs='+')
#p.set_defaults (func=process_train)


#
#class ARGES(object):
#    training_data_src_dir='/home/heimi/myDeepFaceLab/workspace/data2_src/aligned'
#    training_data_dst_dir='/home/heimi/myDeepFaceLab/workspace/data2_dst/aligned'
##    pretraining_data_dir=None
#    model_dir='/home/heimi/myDeepFaceLab/workspace/model64'
#    model_name='H64'
#    no_preview='no_preview'
##    debug=
##    [ [int(x[0]), x[1] ] for x in arguments.execute_program ]
#    
#    
#    
#    
#
#arguments=ARGES()
#process_train(arguments)


args={'training_data_src_dir': '/home/heimi/aaDeepFaceLab/workspace/data2_src/aligned',
 'training_data_dst_dir': '/home/heimi/aaDeepFaceLab/workspace/data2_dst/aligned', 
 'pretraining_data_dir': None, 
 'model_path': '/home/heimi/aaDeepFaceLab/workspace/model2',
 'model_name': 'H64', 'no_preview': False, 
 'debug': False, 
 'execute_programs': []}


device_args={'cpu_only': True,
             'force_gpu_idx': -1}

#if __name__=='__main__':
process_train(args,device_args)



