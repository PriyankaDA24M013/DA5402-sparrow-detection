from model import get_model
import cv2
import torch
import torchvision.transforms as T

transform = T.Compose([T.ToTensor()])

def predict_image(img_path):
    model = get_model(num_classes=2, fine_tune=False)
    model.eval()
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        prediction = model(tensor)
    return prediction
