import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)



# MODEL DEFINATION
# Vgg16 
vgg16_model = models.vgg16(pretrained=False)
num_ftrs_vgg = vgg16_model.classifier[6].in_features
vgg16_model.classifier[6] = nn.Linear(num_ftrs_vgg, 5)

# densenet121 
densenet121_model = models.densenet121(pretrained=False)
num_ftrs_densenet = densenet121_model.classifier.in_features
densenet121_model.classifier = nn.Linear(num_ftrs_densenet, 5)

# resnet18
resnet18_model = models.resnet18(pretrained=True)
num_ftrs_resnet = resnet18_model.fc.in_features
resnet18_model.fc = nn.Linear(num_ftrs_resnet, 5)



# LOADING SAVED MODELS 
vgg16_model.load_state_dict(torch.load('best_model_vgg16.pt', map_location='cpu'))
densenet121_model.load_state_dict(torch.load('best_model.pt', map_location='cpu'))
resnet18_model.load_state_dict(torch.load('resnet18_checkpoint.pt', map_location='cpu'))

vgg16_model.eval()
densenet121_model.eval()
resnet18_model.eval()

def preprocess_image(image):
    # Resize and normalize the image
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if len(image.size) == 2:
        # Convert grayscale image to RGB
        image = image.convert("RGB")

    image_tensor = data_transform(image).unsqueeze(0)
    return image_tensor

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file found'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'})

        image = Image.open(file)
        image_tensor = preprocess_image(image)

        with torch.no_grad():
            vgg16_outputs = vgg16_model(image_tensor)
            densenet121_outputs = densenet121_model(image_tensor)
            resnet18_outputs = resnet18_model(image_tensor)

        _, vgg16_predicted = torch.max(vgg16_outputs, 1)
        _, densenet121_predicted = torch.max(densenet121_outputs, 1)
        _, resnet18_predicted = torch.max(resnet18_outputs, 1)

        labels = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']

        vgg16_predicted_class = labels[vgg16_predicted.item()]
        densenet121_predicted_class = labels[densenet121_predicted.item()]
        resnet18_predicted_class = labels[resnet18_predicted.item()]

        return jsonify({
            'vgg16_prediction': vgg16_predicted_class,
            'densenet121_prediction': densenet121_predicted_class,
            'resnet18_prediction': resnet18_predicted_class
        })

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
