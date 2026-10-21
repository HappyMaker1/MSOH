import numpy as np
from src.distance import dtw_distance


def test_dtw_scalar_identity():
    assert dtw_distance([1,2,3],[1,2,3],1) == 0.0


def test_dtw_vector_identity():
    x=np.array([[0,1],[1,2],[2,3]],float)
    assert dtw_distance(x,x,1) == 0.0
