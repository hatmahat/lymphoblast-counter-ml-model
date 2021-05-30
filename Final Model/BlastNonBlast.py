# Paths
WORKSPACE_PATH = 'Tensorflow/workspace'; print("WORKSPACE_PATH       :" + WORKSPACE_PATH)
SCRIPTS_PATH = 'Tensorflow/scripts'; print("SCRIPTS_PATH         :" + SCRIPTS_PATH)
APIMODEL_PATH = 'Tensorflow/models'; print("APIMODEL_PATH        :" + APIMODEL_PATH)
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'; print("ANNOTATION_PATH      :" + ANNOTATION_PATH)
IMAGE_PATH = WORKSPACE_PATH+'/images'; print("IMAGE_PATH           :" + IMAGE_PATH)
MODEL_PATH = WORKSPACE_PATH+'/models'; print("MODEL_PATH           :" + MODEL_PATH)
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'; print("PRETRAINED_MODEL_PATH:" + PRETRAINED_MODEL_PATH)
CONFIG_PATH = MODEL_PATH+'/my_ssd_resnet50/pipeline.config'; print("CONFIG_PATH          :" + CONFIG_PATH)
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_resnet50/'; print("CHECKPOINT_PATH      :" + CHECKPOINT_PATH)

CUSTOM_MODEL_NAME = 'my_ssd_resnet50' 
# Config for Transfer learning
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# GPU Check
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
#print("-"*30)
#print(physical_devices[0])
#print("-"*30)

#config = tf.compat.v1.ConfigProto(
#        device_count = {'GPU': 0}
#    )
#sess = tf.compat.v1.Session(config=config)

# Hide GPU from visible devices
#tf.config.set_visible_devices([], 'GPU')

'''
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
'''

# 1.5GB

CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'

# Load model from checkpoint
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-59')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

#Detect Image
import cv2 
import numpy as np
import os
from matplotlib import pyplot as plt
#%matplotlib inline

category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')

# read by path
img = cv2.imread("Im010_1.jpg") # baca gambar
image_np = np.array(img)        # ubah ke np
#print(img)
#print(image_np)

input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

label_id_offset = 1
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=20,
            min_score_thresh=.65,
            agnostic_mode=False)

#plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
#plt.show()

# Save image in the same directory
from PIL import Image
im = Image.fromarray(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
im.save("010-1.jpg")