import os
import torch
from torchvision.models import resnet18, ResNet18_Weights, MobileNet_V3_Large_Weights, mobilenet_v3_large, swin_t, Swin_T_Weights, vit_b_16, ViT_B_16_Weights
from model import CustomFCN, CustomCNN, CustomLSTM, Bert, T5, GPTNeo


def get_model_name(model_name: str, device: torch.device, batch_size: int) -> str:
    model = load_model(model_name=model_name, device=device, batch_size=batch_size)
    return model.__class__.__name__


def load_model(model_name: str, device: torch.device, batch_size: int) -> torch.nn.Module:
    if model_name == "swin_t":
        model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    elif model_name == "vit":
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    elif model_name == "resnet":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "mobilenet":
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    elif model_name == "fcn":
        model = CustomFCN(input_size=3*224*224, hidden_size=224, num_classes=1000)
    elif model_name == "cnn":
        model = CustomCNN(num_classes=1000)
    elif model_name == "rnn":
        model = CustomLSTM(
            input_size=224,
            hidden_size=100,
            layer_size=100,
            num_classes=1000,
            batch_size=batch_size,
            device=device,
        )
    elif model_name == "bert":
        model = Bert()
    elif model_name == "t5":
        model = T5()
    elif model_name == "gptneo":
        model = GPTNeo()

    model.eval()
    return model


def save_torchscript_model(
    model: torch.nn.Module,
    model_torchscript_path: str,
    example_inputs = None,
) -> None:
    model_dir = os.path.dirname(model_torchscript_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if example_inputs is not None:
        traced_model = torch.jit.trace(model, example_inputs=example_inputs)  # it doesn't work with empty example_inputs
    else:
        traced_model = torch.jit.script(model, example_inputs=example_inputs)

    torch.jit.save(traced_model, model_torchscript_path)

def load_torchscript_model(model_torchscript_path: str, device: torch.device) -> torch.ScriptModule:
    model = torch.jit.load(model_torchscript_path, map_location=device).eval()
    model = torch.jit.optimize_for_inference(model) # this line is essential: https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html#torch.jit.optimize_for_inference
    return model


def save_torchscript(
    model_name: str,
    device: torch.device,
    batch_size: int,
    model_torchscript_path: str,
    example_inputs = None,
) -> None:
    model = load_model(model_name=model_name, device=device, batch_size=batch_size)
    save_torchscript_model(
        model=model,
        model_torchscript_path=model_torchscript_path,
        example_inputs=example_inputs,
    )
