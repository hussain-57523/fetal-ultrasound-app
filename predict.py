# predict.py

import torch
import torch.nn.functional as F

# This list should match the class order from your training notebook
# It's better to define it here or in a config file than to rely on the training dataset object
CLASS_NAMES = [
    'Fetal abdomen',
    'Fetal brain',
    'Fetal femur',
    'Fetal thorax',
    'Maternal cervix',
    'Other'
]

def make_prediction(model, processed_tensor):
    """
    Takes a model and a preprocessed tensor, returns the predicted class,
    confidence, and all class probabilities.
    """
    model.eval()
    with torch.no_grad():
        # Get the model's raw output (logits)
        logits = model(processed_tensor)
        
        # Convert logits to probabilities using softmax
        probabilities = F.softmax(logits, dim=1)
        
        # Get the top probability and its corresponding class index
        top_p, top_class_idx = probabilities.topk(1, dim=1)
        
        predicted_prob = top_p.item()
        predicted_class_index = top_class_idx.item()
        predicted_class_name = CLASS_NAMES[predicted_class_index]

        # Get all class probabilities for display
        all_probs = {CLASS_NAMES[i]: probabilities[0, i].item() for i in range(len(CLASS_NAMES))}

    return predicted_class_name, predicted_prob, all_probs