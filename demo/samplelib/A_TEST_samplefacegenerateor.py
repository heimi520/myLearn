
import sys
from os import path
sys.path.append( path.dirname(path.dirname(path.abspath(__file__))))


import multiprocessing
import traceback

import cv2
import numpy as np

from facelib import LandmarksProcessor
from samplelib import (SampleGeneratorBase, SampleLoader, SampleProcessor,
                       SampleType)
from utils import iter_utils

from enum import IntEnum

import cv2
import numpy as np

import imagelib
from facelib import FaceType, LandmarksProcessor


'''
arg
output_sample_types = [
                        [SampleProcessor.TypeFlags, size, (optional) {} opts ] ,
                        ...
                      ]
'''
class SampleGeneratorFace(SampleGeneratorBase):
    def __init__ (self, 
                  samples_path,
                  debug, 
                  batch_size,
                  sort_by_yaw=False, 
                  sort_by_yaw_target_samples_path=None,
                  random_ct_samples_path=None, 
                  sample_process_options=SampleProcessor.Options(),
                  output_sample_types=[], 
                  add_sample_idx=False,
                  generators_count=2, 
                  generators_random_seed=None,
                  **kwargs):
        
        super().__init__(samples_path, debug, batch_size)
        self.sample_process_options = sample_process_options
        self.output_sample_types = output_sample_types
        self.add_sample_idx = add_sample_idx

        if sort_by_yaw_target_samples_path is not None:
            self.sample_type = SampleType.FACE_YAW_SORTED_AS_TARGET
        elif sort_by_yaw:
            self.sample_type = SampleType.FACE_YAW_SORTED
        else:
            self.sample_type = SampleType.FACE

        if generators_random_seed is not None and len(generators_random_seed) != generators_count:
            raise ValueError("len(generators_random_seed) != generators_count")

        self.generators_random_seed = generators_random_seed

        samples = SampleLoader.load (self.sample_type, self.samples_path, sort_by_yaw_target_samples_path)
        ct_samples = SampleLoader.load (SampleType.FACE, random_ct_samples_path) if random_ct_samples_path is not None else None
        self.random_ct_sample_chance = 100

        if self.debug:
            self.generators_count = 1
            self.generators = [iter_utils.ThisThreadGenerator ( self.batch_func, (0, samples, ct_samples) )]
        else:
            self.generators_count = min ( generators_count, len(samples) )
            self.generators = [iter_utils.SubprocessGenerator ( self.batch_func, (i, samples[i::self.generators_count], ct_samples ) ) for i in range(self.generators_count) ]

        self.generator_counter = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.generator_counter += 1
        generator = self.generators[self.generator_counter % len(self.generators) ]
        return next(generator)

    def batch_func(self, param ):
        print('batch_func////////param//////',param)
        generator_id, samples, ct_samples = param

        if self.generators_random_seed is not None:
            np.random.seed ( self.generators_random_seed[generator_id] )

        samples_len = len(samples)
        samples_idxs = [*range(samples_len)]

        ct_samples_len = len(ct_samples) if ct_samples is not None else 0

        if len(samples_idxs) == 0:
            raise ValueError('No training data provided.')

        if self.sample_type == SampleType.FACE_YAW_SORTED or self.sample_type == SampleType.FACE_YAW_SORTED_AS_TARGET:
            if all ( [ samples[idx] == None for idx in samples_idxs] ):
                raise ValueError('Not enough training data. Gather more faces!')

        if self.sample_type == SampleType.FACE:
            shuffle_idxs = []
        elif self.sample_type == SampleType.FACE_YAW_SORTED or self.sample_type == SampleType.FACE_YAW_SORTED_AS_TARGET:
            shuffle_idxs = []
            shuffle_idxs_2D = [[]]*samples_len

        while True:
            batches = None
            for n_batch in range(self.batch_size):
                while True:
                    sample = None
                    if self.sample_type == SampleType.FACE:
                        if len(shuffle_idxs) == 0:
                            shuffle_idxs = samples_idxs.copy()
                            np.random.shuffle(shuffle_idxs)
                        
                        print('shuffle_idx len///',len(shuffle_idxs))
                        idx = shuffle_idxs.pop()
                        sample = samples[ idx ]
                        print('shuffle_idx len222///',len(shuffle_idxs))
                        print('sample',sample)

                    elif self.sample_type == SampleType.FACE_YAW_SORTED or self.sample_type == SampleType.FACE_YAW_SORTED_AS_TARGET:
                        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                        if len(shuffle_idxs) == 0:
                            shuffle_idxs = samples_idxs.copy()
                            np.random.shuffle(shuffle_idxs)

                        idx = shuffle_idxs.pop()
                        if samples[idx] != None:
                            if len(shuffle_idxs_2D[idx]) == 0:
                                a = shuffle_idxs_2D[idx] = [ *range(len(samples[idx])) ]
                                np.random.shuffle (a)

                            idx2 = shuffle_idxs_2D[idx].pop()
                            sample = samples[idx][idx2]

                            idx = (idx << 16) | (idx2 & 0xFFFF)

                    if sample is not None:
                        try:
                            ct_sample=None                            
                            if ct_samples is not None:                                
                                if np.random.randint(100) < self.random_ct_sample_chance:
                                    ct_sample=ct_samples[np.random.randint(ct_samples_len)]
                            print('sample here111',sample)
                            print('self.sample_process_options',self.sample_process_options)
                            print('self.output_sample_types',self.output_sample_types)
                            print('debug',self.debug)
                            print('ct_ample',ct_sample)
                            x = SampleProcessor.process (sample, self.sample_process_options, self.output_sample_types, self.debug, ct_sample=ct_sample)
                            print('x/////',x)
                        except:
                            raise Exception ("Exception occured in sample %s. Error: %s" % (sample.filename, traceback.format_exc() ) )

                        if type(x) != tuple and type(x) != list:
                            raise Exception('SampleProcessor.process returns NOT tuple/list')

                        if batches is None:
                            batches = [ [] for _ in range(len(x)) ]
                            if self.add_sample_idx:
                                batches += [ [] ]
                                i_sample_idx = len(batches)-1
                        print('batchtest///',batches)
                        for i in range(len(x)):
                            batches[i].append ( x[i] )

                        if self.add_sample_idx:
                            batches[i_sample_idx].append (idx)

                        break
#            yield [ np.array(batch) for batch in batches]
            yield batches








#
#SampleGeneratorFace(self.training_data_src_path, 
#                    sort_by_yaw_target_samples_path=self.training_data_dst_path if self.sort_by_yaw else None,
#                        debug=self.is_debug(), 
#                        batch_size=self.batch_size,
#                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip, scale_range=np.array([-0.05, 0.05])+self.src_scale_mod / 100.0 ),
#                        output_sample_types=output_sample_types),
#


#samples = SampleLoader.load (SampleType.FACE, '/home/heimi/myDeepFaceLab/workspace/data2_src/aligned', None)




t = SampleProcessor.Types
output_sample_types=[ { 'types': (t.IMG_WARPED_TRANSFORMED, t.FACE_TYPE_HALF, t.MODE_BGR), 'resolution':64},
                  { 'types': (t.IMG_TRANSFORMED, t.FACE_TYPE_HALF, t.MODE_BGR), 'resolution':64},
                  { 'types': (t.IMG_TRANSFORMED, t.FACE_TYPE_HALF, t.MODE_M), 'resolution':64} ]




samples = SampleLoader.load (SampleType.FACE, '/home/heimi/aaDeepFaceLab/workspace/data2_src/aligned', None)
print('sample len///',len(samples))     
sample=samples[1]
sample_bgr = sample.load_bgr()
img_test=(sample_bgr*255).astype('uint8')
#import cv2    #导入opencv


img = cv2.imread(sample.filename)      #利用imread()读入图像，将图像存入到img中，类型为numpu.ndarray
cv2.imwrite('test.png',img)  #这样就可以看到彩色原图了



sp=SampleProcessor()

x = sp.process (sample, 
                 SampleProcessor.Options(random_flip=True, scale_range=np.array([-0.05, 0.05])+50/ 100.0 ),
#                   
#                             SampleProcessor.Options(), 
                 output_sample_types, 
                 False, 
                 ct_sample=None
                 )






sample_process_options=SampleProcessor.Options(random_flip=True, scale_range=np.array([-0.05, 0.05])+50/ 100.0 )



class Types(IntEnum):
    NONE = 0

    IMG_TYPE_BEGIN = 1
    IMG_SOURCE                     = 1
    IMG_WARPED                     = 2
    IMG_WARPED_TRANSFORMED         = 3
    IMG_TRANSFORMED                = 4
    IMG_LANDMARKS_ARRAY            = 5 #currently unused
    IMG_PITCH_YAW_ROLL             = 6
    IMG_PITCH_YAW_ROLL_SIGMOID     = 7
    IMG_TYPE_END = 10

    FACE_TYPE_BEGIN = 10
    FACE_TYPE_HALF             = 10
    FACE_TYPE_FULL             = 11
    FACE_TYPE_HEAD             = 12  #currently unused
    FACE_TYPE_AVATAR           = 13  #currently unused
    FACE_TYPE_END = 20

    MODE_BEGIN = 40
    MODE_BGR                   = 40  #BGR
    MODE_G                     = 41  #Grayscale
    MODE_GGG                   = 42  #3xGrayscale
    MODE_M                     = 43  #mask only
    MODE_BGR_SHUFFLE           = 44  #BGR shuffle
    MODE_END = 50

class Options(object):

    def __init__(self, random_flip = True, rotation_range=[-10,10], scale_range=[-0.05, 0.05], tx_range=[-0.05, 0.05], ty_range=[-0.05, 0.05] ):
        self.random_flip = random_flip
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.tx_range = tx_range
        self.ty_range = ty_range


import collections

debug=False
#def process (sample, sample_process_options, output_sample_types, debug, ct_sample=None):
SPTF = SampleProcessor.Types

sample_bgr = sample.load_bgr()
ct_sample_bgr = None
ct_sample_mask = None
h,w,c = sample_bgr.shape

is_face_sample = sample.landmarks is not None

if debug and is_face_sample:
    LandmarksProcessor.draw_landmarks (sample_bgr, sample.landmarks, (0, 1, 0))

params = imagelib.gen_warp_params(sample_bgr, 
                                  sample_process_options.random_flip,
                                  rotation_range=sample_process_options.rotation_range,
                                  scale_range=sample_process_options.scale_range, 
                                  tx_range=sample_process_options.tx_range, 
                                  ty_range=sample_process_options.ty_range )

cached_images = collections.defaultdict(dict)

sample_rnd_seed = np.random.randint(0x80000000)

SPTF_FACETYPE_TO_FACETYPE =  {  SPTF.FACE_TYPE_HALF : FaceType.HALF,
                                SPTF.FACE_TYPE_FULL : FaceType.FULL,
                                SPTF.FACE_TYPE_HEAD : FaceType.HEAD,
                                SPTF.FACE_TYPE_AVATAR : FaceType.AVATAR }

outputs = []
for opts in output_sample_types:
    
    resolution = opts.get('resolution', 0)
    types = opts.get('types', [] )

    random_sub_res = opts.get('random_sub_res', 0)
    normalize_std_dev = opts.get('normalize_std_dev', False)
    normalize_vgg = opts.get('normalize_vgg', False)
    motion_blur = opts.get('motion_blur', None)
    apply_ct = opts.get('apply_ct', False)
    normalize_tanh = opts.get('normalize_tanh', False)

    img_type = SPTF.NONE
    target_face_type = SPTF.NONE
    face_mask_type = SPTF.NONE
    mode_type = SPTF.NONE
    for t in types:
        print(t)
        if t >= SPTF.IMG_TYPE_BEGIN and t < SPTF.IMG_TYPE_END:
            img_type = t
        elif t >= SPTF.FACE_TYPE_BEGIN and t < SPTF.FACE_TYPE_END:
            target_face_type = t
        elif t >= SPTF.MODE_BEGIN and t < SPTF.MODE_END:
            mode_type = t

    if img_type == SPTF.NONE:
        raise ValueError ('expected IMG_ type')

    if img_type == SPTF.IMG_LANDMARKS_ARRAY:
        l = sample.landmarks
        l = np.concatenate ( [ np.expand_dims(l[:,0] / w,-1), np.expand_dims(l[:,1] / h,-1) ], -1 )
        l = np.clip(l, 0.0, 1.0)
        img = l
    elif img_type == SPTF.IMG_PITCH_YAW_ROLL or img_type == SPTF.IMG_PITCH_YAW_ROLL_SIGMOID:
        pitch_yaw_roll = sample.pitch_yaw_roll
        if pitch_yaw_roll is not None:
            pitch, yaw, roll = pitch_yaw_roll
        else:
            pitch, yaw, roll = LandmarksProcessor.estimate_pitch_yaw_roll (sample.landmarks)
        if params['flip']:
            yaw = -yaw

        if img_type == SPTF.IMG_PITCH_YAW_ROLL_SIGMOID:
            pitch = (pitch+1.0) / 2.0
            yaw = (yaw+1.0) / 2.0
            roll = (roll+1.0) / 2.0

        img = (pitch, yaw, roll)
    else:
        if mode_type == SPTF.NONE:
            raise ValueError ('expected MODE_ type')

        img = cached_images.get(img_type, None)
        
            
        if img is None:

            img = sample_bgr
            
#            a=np.zeros_like(img)
#            a[:,:,[1,2]]=img[:,:,[2,1]]
#            
            cv2.imwrite('test_img.png',(img*255).astype('uint8')) 
        
        
            mask = None
            cur_sample = sample

            if is_face_sample:
                if motion_blur is not None:
                    chance, mb_range = motion_blur
                    chance = np.clip(chance, 0, 100)

                    if np.random.randint(100) < chance:
                        mb_range = [3,5,7,9][ : np.clip(mb_range, 0, 3)+1 ]
                        dim = mb_range[ np.random.randint(len(mb_range) ) ]
                        img = imagelib.LinearMotionBlur (img, dim, np.random.randint(180) )

                mask = cur_sample.load_fanseg_mask() #using fanseg_mask if exist
#                cv2.imwrite('test_mask.png',(mask*255).astype('uint8')) 
#        
                if mask is None:
                    mask = LandmarksProcessor.get_image_hull_mask (img.shape, cur_sample.landmarks)
                    cv2.imwrite('test_mask.png',(mask*255).astype('uint8')) 
#        
                if cur_sample.ie_polys is not None:
                    cur_sample.ie_polys.overlay_mask(mask)
                    
                    cv2.imwrite('test_mask2.png',(mask*255).astype('uint8')) 
#        

            warp = (img_type==SPTF.IMG_WARPED or img_type==SPTF.IMG_WARPED_TRANSFORMED)
            transform = (img_type==SPTF.IMG_WARPED_TRANSFORMED or img_type==SPTF.IMG_TRANSFORMED)
            flip = img_type != SPTF.IMG_WARPED

#            img = imagelib.warp_by_params (params, img, warp, transform, flip, True)
            img_1 = imagelib.warp_by_params (params, img, warp, transform, flip, True)
            cv2.imwrite('test_warp_by_params.png',(img_1*255).astype('uint8')) 
#           
            if mask is not None:
                mask = imagelib.warp_by_params (params, mask, warp, transform, flip, False)[...,np.newaxis]
                img = np.concatenate( (img, mask ), -1 )
                
                cv2.imwrite('test_img_mask.png',(img[:,:,[0,1,2,3]]*255).astype('uint8')) 
        

            cached_images[img_type] = img

        if is_face_sample and target_face_type != SPTF.NONE:
            ft = SPTF_FACETYPE_TO_FACETYPE[target_face_type]
            if ft > sample.face_type:
                raise Exception ('sample %s type %s does not match model requirement %s. Consider extract necessary type of faces.' % (sample.filename, sample.face_type, ft) )
#            img = cv2.warpAffine( img, 
#                                 LandmarksProcessor.get_transform_mat (sample.landmarks, resolution, ft), 
#                                 (resolution,resolution), flags=cv2.INTER_CUBIC )
            img2 = cv2.warpAffine( img, 
                                 LandmarksProcessor.get_transform_mat (sample.landmarks, resolution, ft), 
                                 (resolution,resolution), flags=cv2.INTER_CUBIC )
            
            
#            image1=(img*255).astype('uint8')
#            cv2.imwrite('test1.png',image1) 
            
            image2=(img2*255).astype('uint8')
            cv2.imwrite('test_warpaffine.png',image2) 
            
            
            
            
        else:
            img3 = cv2.resize( img, (resolution,resolution), cv2.INTER_CUBIC )
            cv2.imwrite('test_resize.png',(img3*255).astype('uint8')) 
            

        if random_sub_res != 0:
            sub_size = resolution - random_sub_res
            rnd_state = np.random.RandomState (sample_rnd_seed+random_sub_res)
            start_x = rnd_state.randint(sub_size+1)
            start_y = rnd_state.randint(sub_size+1)
            img_random = img[start_y:start_y+sub_size,start_x:start_x+sub_size,:]
            cv2.imwrite('test_random.png',(img_random*255).astype('uint8')) 
            
        
        img_clip = np.clip(img, 0, 1)
        cv2.imwrite('test_img_clip.png',(img_clip*255).astype('uint8')) 
            
        img_bgr  = img_clip[...,0:3]
        img_mask = img_clip[...,3:4]
        
        
        cv2.imwrite('test_img_bgr.png',(img_bgr*255).astype('uint8')) 
        cv2.imwrite('test_img_mask.png',(img_mask*255).astype('uint8')) 
        
#        img = np.clip(img, 0, 1)
#        img_bgr  = img[...,0:3]
#        img_mask = img[...,3:4]

        if apply_ct and ct_sample is not None:
            if ct_sample_bgr is None:
                ct_sample_bgr = ct_sample.load_bgr()

            ct_sample_bgr_resized = cv2.resize( ct_sample_bgr, (resolution,resolution), cv2.INTER_LINEAR )

            img_bgr = imagelib.linear_color_transfer (img_bgr, ct_sample_bgr_resized)
            img_bgr = np.clip( img_bgr, 0.0, 1.0)

        if normalize_std_dev:
            img_bgr = (img_bgr - img_bgr.mean( (0,1)) ) / img_bgr.std( (0,1) )
        elif normalize_vgg:
            img_bgr = np.clip(img_bgr*255, 0, 255)
            img_bgr[:,:,0] -= 103.939
            img_bgr[:,:,1] -= 116.779
            img_bgr[:,:,2] -= 123.68

        if mode_type == SPTF.MODE_BGR:
            img = img_bgr
        elif mode_type == SPTF.MODE_BGR_SHUFFLE:
            rnd_state = np.random.RandomState (sample_rnd_seed)
            img = np.take (img_bgr, rnd_state.permutation(img_bgr.shape[-1]), axis=-1)
            
            img_shuffle = np.take (img_bgr, rnd_state.permutation(img_bgr.shape[-1]), axis=-1)
            cv2.imwrite('test_img_shuffle.png',(img_shuffle*255).astype('uint8')) 
        
            
            
        elif mode_type == SPTF.MODE_G:
            img = np.concatenate ( (np.expand_dims(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),-1),img_mask) , -1 )
        elif mode_type == SPTF.MODE_GGG:
            img = np.concatenate ( ( np.repeat ( np.expand_dims(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),-1), (3,), -1), img_mask), -1)
        elif mode_type == SPTF.MODE_M and is_face_sample:
            img = img_mask

        if not debug:
            if normalize_tanh:
                img = np.clip (img * 2.0 - 1.0, -1.0, 1.0)
            else:
                img = np.clip (img, 0.0, 1.0)

    outputs.append ( img )

if debug:
    result = []

    for output in outputs:
        if output.shape[2] < 4:
            result += [output,]
        elif output.shape[2] == 4:
            result += [output[...,0:3]*output[...,3:4],]

#    return result
#else:
#    return outputs










































########################
#x = SampleProcessor.process (sample, 
#                             SampleProcessor.Options(random_flip=True, scale_range=np.array([-0.05, 0.05])+50/ 100.0 ),
##                   
##                             SampleProcessor.Options(), 
#                             output_sample_types, 
#                             False, 
#                             ct_sample=None
#                             )


#img2=(x[0]*255).astype('uint8')
#
#cv2.imwrite('test2.png',img2)  #这样就可以看到彩色原图了


#print('x/////',x)
#
########################################################333
#sample_iter=SampleGeneratorFace(samples_path='/home/heimi/myDeepFaceLab/workspace/data2_src/aligned',
#                  debug=False, 
#                  batch_size=4,
#                  sort_by_yaw=False, 
#                  sort_by_yaw_target_samples_path=None,
#                  random_ct_samples_path=None, 
##                  sample_process_options=SampleProcessor.Options() ,###,SampleProcessor.Options(random_flip=self.random_flip, scale_range=np.array([-0.05, 0.05])+self.src_scale_mod / 100.0 ),
#                  sample_process_options=SampleProcessor.Options(random_flip=True, scale_range=np.array([-0.05, 0.05])+0 / 100.0 ),
#                   
#                  output_sample_types=output_sample_types, 
#                  add_sample_idx=False,
#                  generators_count=2, 
#                  generators_random_seed=None,
#                  )
#
##
#aa=next(sample_iter)
#
#a1=aa[0]
#a11=a1[0]
#

#############################################################
#for k,tmp in enumerate(aa):
#    print('k',k,'shape',len(tmp))
#    for kk,b in  enumerate(tmp):
#        b=(b*255).astype('uint8')
#        print(k,kk,b.shape)
#        cv2.imwrite('test_%s_%s.png'%(k,kk),b)  #这样就可以看到彩色原图了






