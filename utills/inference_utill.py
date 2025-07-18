from models import get_model
from torchvision import transforms
import torch
from PIL import Image
import cv2


def load_model(model_path, config):
    model = get_model(config)
    weight = torch.load(model_path) if config['device'] == 'cuda:0' else (torch.load(model_path, map_location='cpu'))
    # model.load_state_dict(weight['model'],  strict=False)
    model.load_state_dict(weight, strict=False)
    model.eval()
    return model


def build_eval_transformation(config):
    return transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize((config['model_width'], config['model_height']), transforms.InterpolationMode.BICUBIC),
        transforms.Normalize(mean=config['mean'], std=config['std']),
    ])


def predict(pil_image, transformation, model, device):
    image = transformation(pil_image).unsqueeze(0)
    if device == 'cuda:0':
        image = image.to(device)
        model.to(device)
    outputs = model(image)
    if isinstance(outputs, dict):
        probabilities = {}
        for i, out in enumerate(outputs):
            probabilities[out] = torch.nn.Softmax(dim=-1)(outputs[out])
    else:
        probabilities = torch.nn.Softmax(dim=-1)(outputs)
    return probabilities


def break_image_in_four_parts(image):
    width = image.shape[1]
    width_cutoff = width // 2
    l1 = image[:, :width_cutoff]
    l2 = image[:, width_cutoff:]
    height_cutoff_l1 = l1.shape[0] // 2
    height_cutoff_l2 = l2.shape[0] // 2
    first, second = l1[:height_cutoff_l1, :], l1[height_cutoff_l1:, :]
    third, fourth = l2[:height_cutoff_l2, :], l2[height_cutoff_l2:, :]
    return Image.fromarray(cv2.cvtColor(first, cv2.COLOR_BGR2RGB)), Image.fromarray(
        cv2.cvtColor(second, cv2.COLOR_BGR2RGB)), Image.fromarray(
        cv2.cvtColor(third, cv2.COLOR_BGR2RGB)), Image.fromarray(cv2.cvtColor(fourth, cv2.COLOR_BGR2RGB))


def postprocess_output(probabilities, classes):
    if isinstance(probabilities, dict):
        prediction_indexes = {}
        final_prediction = {}
        for key in probabilities.keys():
            _, prediction_indexes[key] = torch.max(probabilities[key], 1)

            final_prediction[key] = classes[key][prediction_indexes[key].item()]
        return final_prediction
    else:
        _, prediction_index = torch.max(probabilities, 1)
        return classes[prediction_index.item()]