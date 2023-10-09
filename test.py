import numpy as np

def numpy_multinomial(probs, num_samples, replacement=True):
    """
    Simulate torch.multinomial using NumPy.

    Args:
    - probs (ndarray): An array representing the probability distribution.
    - num_samples (int): The number of samples to draw.
    - replacement (bool): Whether to sample with replacement.

    Returns:
    - ndarray: An array of sampled indices.
    """
    if not replacement and num_samples > len(probs):
        raise ValueError("num_samples must be less than or equal to the number of elements in probs when replacement is False.")

    cumulative_probs = np.cumsum(probs)
    sampled_indices = []

    for _ in range(num_samples):
        rand_val = np.random.rand()
        index = np.searchsorted(cumulative_probs, rand_val)
        sampled_indices.append(index)

        if not replacement:
            cumulative_probs[index:] -= probs[index]
    
    return np.array(sampled_indices, dtype=np.int32)

# Example usage:
probs = np.array([0.6, 0.1, 0.2, 0.1])  # Example probabilities
num_samples = 1
samples = numpy_multinomial(probs, num_samples)
print("Sampled indices:", samples)
