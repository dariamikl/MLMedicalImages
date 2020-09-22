from pathlib import Path

import cv2
import torch
import numpy as np

from model.model import get_preprocessing2, ModelClassification, read_png

CHECKPOINT_PATHS = ['weights/ckpt.ckpt', 'weights/kaggle_chpt.ckpt']
THRESHOLDS = [0.385, 0.285, 0.345, 0.25, 0.425, 0.22, 0.28, 0.335, 0.36, 0.38, 0.33, 0.405, 0.42, 0.235]
idx2label = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
             'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
             'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

IMG_PATH = '/home/vladbakhteev/data/hack/images/1000155.png'


def predict(img, models, preprocessing, device):
    img_tensor = preprocessing(image=img)['image'].float().unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        probs = torch.stack([model(img_tensor)[0].sigmoid().cpu() for model in models])
        probs = probs.mean(dim=0)
    preds = (probs > torch.tensor(THRESHOLDS)).int().numpy()
    labels = list(np.nonzero(preds)[0])
    labels = [idx2label[idx] for idx in labels]
    return probs.numpy(), labels


def get_cam(img, model, preprocessing, device):
    backbone = model.backbone
    head = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        model.head
    )
    
    img_tensor = preprocessing(image=img)['image'].float().unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    
    with torch.no_grad():
        _, _, _, _, feature_map = backbone(img_tensor)
    feature_map.requires_grad = True
    logits = head(feature_map)[0]
    
    label = logits.argmax()
    class_score = logits[label]
    class_score.backward()
    
    grad = feature_map.grad
    c = torch.nn.AdaptiveAvgPool2d((1, 1))(grad)
    cam = torch.clamp((c * feature_map).sum(1), min=0)
    
    return cam.detach().squeeze().cpu().numpy()

def apply_cam(img, model, preprocessing, device):
    cam = get_cam(img, model, preprocessing, device)
    cam = ((cam / cam.max()) * 255).astype(np.uint8)
    cam = cv2.resize(cam, img.shape[:2][::-1])

    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    result = (heatmap * 0.4 + img * 0.6).astype(np.uint8)
    return result


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = []
    for ch_p in CHECKPOINT_PATHS:
        model = ModelClassification.load_from_checkpoint(ch_p).model
        model.to(device)
        model.eval()
        models.append(model)

    preprocessing = get_preprocessing2(size=256, model_name='resnet50', num_channels=1)

    img = read_png(IMG_PATH)[:, :, None]  # (h, w, 1)
    if len(img.shape) > 2:
        img = img[:, :, 0]
    img = img[:, :, None]
    print(img.shape)
    probs, labels = predict(img, models, preprocessing, device)

    if len(labels) > 0:
        result_img = apply_cam(img, models[0], preprocessing, device)[:, :, ::-1]
    else:
        result_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    print(labels) # Response for orginizers
    result_img        # RGB image for interface
    return labels, result_img


if __name__=='__main__':
    main()