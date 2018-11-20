import torch
import numpy as np
from torchvision import datasets
from data import data_transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2

from model import Net
state_dict = torch.load('experiment/model_7.pth')
model = Net()
model.load_state_dict(state_dict)

features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

model.eval()
model.model._modules.get('layer4').register_forward_hook(hook_feature)

weights = list(model.model.fc.parameters())[0].data.numpy()
print(weights.shape)


def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        #cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam
  
import PIL.Image as Image
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

from tqdm import tqdm
l = 0
for fold in tqdm(os.listdir('bird_dataset/train_images')):
  #os.mkdir('bird_dataset/superextratrain_images/' + fold)
  for file in tqdm(os.listdir('bird_dataset/train_images/' + fold)):
    if 'jpg' in file:
      im = (pil_loader('bird_dataset/train_images/' + fold+ '/' + file))
      #print(file)
      #f, (ax1, ax2) = plt.subplots(1, 2)
      #ax1.imshow(im)
      
      
      
      data = data_transforms['test'](im)
      data = data.view(1, data.size(0), data.size(1), data.size(2))
 
      output = model(data)
      pred = output.data.max(1, keepdim=True)[1]
     
      #print(pred, l)
      #break

      CAMs = returnCAM(features_blobs[0], weights, [pred])
      CAMs[0] = cv2.resize(CAMs[0], im.size)
      
      top = 0
      print(np.max((CAMs[0])[top, :]))
      while np.max((CAMs[0])[top, :]) <= 0.5:
        top+=1

      bottom = CAMs[0].shape[0] - 1
      while np.max((CAMs[0])[bottom, :]) <= 0.5:
         bottom-=1
  
      left = 0
      while np.max((CAMs[0])[:, left]) <= 0.5:
        left+=1
      right = CAMs[0].shape[1] - 1
      while np.max((CAMs[0])[:, right]) <= 0.5:
        right-=1
      #print('here')
      augmented = Image.fromarray(np.array(im)[top:bottom, left:right, :])
      augmented.save('bird_dataset/train_images/' + fold+ '/extra' + file)
      features_blobs.clear()
 
      
