import torch
import torch.nn.functional as F

CLASS_NAMES = ['Fetal abdomen', 'Fetal brain', 'Fetal femur', 'Fetal thorax', 'Maternal cervix', 'Other']

def make_prediction(model, processed_tensor):
    model.eval()
    with torch.no_grad():
        logits = model(processed_tensor)
        probabilities = F.softmax(logits, dim=1)
        all_probs = {CLASS_NAMES[i]: probabilities[0, i].item() for i in range(len(CLASS_NAMES))}
        confidence, top_class_idx = torch.max(probabilities, 1)
        predicted_class_name = CLASS_NAMES[top_class_idx.item()]
    return predicted_class_name, confidence.item(), all_probs