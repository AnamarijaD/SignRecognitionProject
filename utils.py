

def adjust_class_labels(label):
    """Subtracts one from the label when greater than 9."""
    if label >= 10:
        label -= 1
    
    return label