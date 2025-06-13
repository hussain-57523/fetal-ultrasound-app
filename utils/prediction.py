# utils/prediction.py

import torch
import torch.nn.functional as F

CLASS_NAMES = [
    'Fetal abdomen', 'Fetal brain', 'Fetal femur', 
    'Fetal thorax', 'Maternal cervix', 'Other'
]

def make_prediction(model, processed_tensor):
    """
    Performs inference and returns prediction details.
    """
    model.eval()
    with torch.no_grad():
        logits = model(processed_tensor)
        probabilities = F.softmax(logits, dim=1)
        top_p, top_class_idx = probabilities.topk(1, dim=1)
        
        predicted_prob = top_p.item()
        predicted_class_index = top_class_idx.item()
        predicted_class_name = CLASS_NAMES[predicted_class_index]
    
    return predicted_class_name, predicted_prob