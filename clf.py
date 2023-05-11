from torchvision.models import resnet18, ResNet18_Weights
from torchvision import models, transforms
import numpy as np
import torch
import cv2
from PIL import Image

IMAGE_SIZE = (256, 256)


def segment_img(img):
    img = cv2.resize(img, IMAGE_SIZE)

    ###### GRAYSCALING ######
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    

    ###### MORPHOLOGICAL FILTERING ######
    kernel = np.ones((5,5), 'uint8')
    #dilation
    dil = cv2.dilate(gray, kernel, iterations=1)
    #erosion
    ero = cv2.erode(dil, kernel)
    
    ########### THRESHOLDING ############
    blur = cv2.GaussianBlur(ero,(7,7),0)
    thresh = cv2.threshold(blur, 0, 255, 
                            cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

    ########## EDGE DETECTION ##########
    blur1 = cv2.GaussianBlur(thresh,(5,5),0) 
    edges = cv2.Canny(image=blur1, threshold1=100, threshold2=200)
    #edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)

    ###### MASKING AND SEGMENTING #######
    cnt = sorted(cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
    mask = np.zeros(IMAGE_SIZE, np.uint8)
    masked = cv2.drawContours(mask, [cnt],-1, 255, -1)
    dst = cv2.bitwise_and(img, img, mask=mask)
    segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    segimg = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
    return segimg


def predict(image_path):
    
    image_path = cv2.resize(image_path, IMAGE_SIZE)
    image_path = segment_img(image_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    model_save_name = '_classifier_.pth'
    path = F"/Users/bryllejames/Desktop/streamlit/{model_save_name}"
    model = torch.load(path, map_location=torch.device('cpu'))
    model.to(device)
    
    
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )])

    
    img = Image.fromarray(image_path.astype('uint8'), 'RGB')
    batch_t = torch.unsqueeze(transform(img), 0)
    batch_t = batch_t.to(device)

    model.eval()
    out = model(batch_t)

    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    with open('description_classes.txt') as f:
        des_classes = [line.strip() for line in f.readlines()]
    

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item(), des_classes[idx]) for idx in indices[0][:4]]
