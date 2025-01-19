def calculate_confidence(probability):
    """Calculate confidence score based on distance from decision boundary"""
    distance_from_boundary = abs(probability - 0.5)
    confidence = distance_from_boundary * 2 * 100  # Scale to percentage
    return min(100, confidence)