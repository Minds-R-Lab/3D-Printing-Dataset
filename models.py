# fault_detection_pipeline/models.py
import torch
import torch.nn as nn
from torchvision import models
import config

def get_model(model_name, num_classes, pretrained=True):
    """
    Loads a pre-trained model from torchvision and modifies its final layer
    for the specified number of classes.

    Args:
        model_name (str): Name of the model (e.g., 'resnet18', 'vgg16').
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to load pre-trained weights.

    Returns:
        torch.nn.Module: The modified model.
    """
    model = None
    weights = models.get_model_weights(model_name).DEFAULT if pretrained else None

    if model_name == 'resnet18':
        model = models.resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'resnet34':
        model = models.resnet34(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(weights=weights)
        # VGG's classifier is a Sequential module
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=weights)
        # MobileNetV2's classifier is also a Sequential module
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'efficientnet_b0':
         model = models.efficientnet_b0(weights=weights)
         num_ftrs = model.classifier[1].in_features
         model.classifier[1] = nn.Linear(num_ftrs, num_classes)



    # --- NEW MODELS ---
    elif model_name == 'resnet50':
        model = models.resnet50(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'densenet121':
        model = models.densenet121(weights=weights)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'efficientnet_b4':
        model = models.efficientnet_b4(weights=weights)
        num_ftrs = model.classifier[1].in_features # EfficientNets usually have Sequential classifier
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'vit_b_16': # Vision Transformer Base (16x16 patches)
        model = models.vit_b_16(weights=weights)
        # ViT uses 'heads' structure, classifier is often 'head' inside 'heads'
        # Verify structure if error occurs: print(model) after loading
        try:
            num_ftrs = model.heads.head.in_features
            model.heads.head = nn.Linear(num_ftrs, num_classes)
        except AttributeError:
             # Fallback or alternative structure check if needed
             print("Could not find model.heads.head, checking model.head")
             num_ftrs = model.head.in_features
             model.head = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'swin_t': # Swin Transformer Tiny
        model = models.swin_t(weights=weights)
        # Swin Transformer typically uses 'head' for the final layer
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, num_classes)
    # --- END NEW MODELS ---

    # Add more models here if needed
    # elif model_name == 'your_custom_model':
    #     model = YourCustomModel(num_classes=num_classes)
    else:
        raise ValueError(f"Model '{model_name}' not supported yet.")

    print(f"Loaded model: {model_name} {'(pretrained)' if pretrained else ''}")
    return model

if __name__ == '__main__':
    # Test loading a model
    test_model_name = config.MODEL_NAMES[0] # Get the first model from config
    print(f"Testing model loading for: {test_model_name}")
    try:
        model = get_model(test_model_name, config.NUM_CLASSES, pretrained=True)
        print(f"Successfully loaded {test_model_name}")
        # print(model) # Uncomment to see model structure

        # Test forward pass with dummy data
        dummy_input = torch.randn(config.BATCH_SIZE, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        output = model(dummy_input)
        print(f"Output shape for a batch: {output.shape}") # Should be [batch_size, num_classes]
        print("Model loading and forward pass test successful!")

    except Exception as e:
        print(f"Error loading model {test_model_name}: {e}")