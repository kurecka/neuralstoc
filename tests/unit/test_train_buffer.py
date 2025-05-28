import numpy as np
from neuralstoc.rsm.train_buffer import TrainBuffer
    

def test_train_buffer():
    """Test the TrainBuffer class"""
    buffer = TrainBuffer(max_size=10)
    buffer.append(np.random.randn(3, 2))
    assert len(buffer) == 3
    buffer.append(np.random.randn(1, 2))
    assert len(buffer) == 4
    buffer.extend([np.random.randn(2, 2), np.random.randn(3, 2)])
    assert len(buffer) == 9
    buffer.append(np.arange(10).reshape(5, 2))
    assert len(buffer) == 10
