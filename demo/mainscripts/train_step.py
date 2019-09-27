
#
import sys
from os import path
sys.path.append( path.dirname(path.dirname(path.abspath(__file__))))


#import sys
#import traceback
#import queue
#import threading
#import time
import numpy as np
#import itertools
from pathlib import Path
#from utils import Path_utils
#import imagelib
#import cv2
import models
#from interact import interact as io




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
#process_train(args,device_args)


training_data_src_path = Path( args.get('training_data_src_dir', '') )
training_data_dst_path = Path( args.get('training_data_dst_dir', '') )

pretraining_data_path = args.get('pretraining_data_dir', '')
pretraining_data_path = Path(pretraining_data_path) if pretraining_data_path is not None else None

model_path = Path( args.get('model_path', '') )
model_name = args.get('model_name', '')
save_interval_min = 15
debug = args.get('debug', '')
execute_programs = args.get('execute_programs', [])


model = models.import_model(model_name)(
            model_path,
            training_data_src_path=training_data_src_path,
            training_data_dst_path=training_data_dst_path,
            pretraining_data_path=pretraining_data_path,
            debug=debug,
            device_args=device_args)

#
iter, iter_time = model.train_one_iter()
#
loss_history = model.get_loss_history()






#def send_preview():
#    if not debug:
#        previews = model.get_previews()
##                    c2s.put ( {'op':'show', 'previews': previews, 'iter':model.get_iter(), 'loss_history': model.get_loss_history().copy() } )
#    else:
#        previews = [( 'debug, press update for new', model.debug_one_iter())]
##                    c2s.put ( {'op':'show', 'previews': previews} )
#
##model_save()
#send_preview()
#
#
#        print('here222222222222222')
#model.pass_one_iter()
#send_preview()

#             
