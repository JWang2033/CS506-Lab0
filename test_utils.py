## Please fill in all the parts labeled as ### YOUR CODE HERE

import numpy as np
import pytest
from utils import *

def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = dot_product(vector1, vector2)
    
    assert result == 32, f"Expected 32, but got {result}"
    
def test_cosine_similarity():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    # Calculate cosine similarity
    dot_product = np.dot(vector1, vector2)
    
    result = dot_product / (norm1 * norm2)
    
    expected_result = 0.9746318461970762
    
    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_nearest_neighbor():
    # Example data
    dataset = np.array([[1, 2], [3, 4], [5, 6]])
    query_point = np.array([2, 3])
    distances = np.linalg.norm(dataset - query_point, axis=1)

    # Calculate distances (Euclidean distance)
    
    result = np.argmin(distances)
    
    expected_index = 0
    
    assert result == expected_index, f"Expected index {expected_index}, but got {result}"
