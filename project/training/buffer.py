# fileName: training/buffer.py
import random
from collections import deque

class ExperienceReplayBuffer:
    def __init__(self, capacity=256):
        """
        capacity: 存储的完整序列的最大数量。
        注意：由于我们现在存储的是完整序列，容量应该比存储单token时小以节省内存。
        """
        self.capacity = capacity
        # 用于序列级数据的统一池
        self.pool = deque(maxlen=capacity)

    def add(self, feedback_samples):
        """
        feedback_samples: 包含序列级字典的列表
        格式：[{"input_ids": ..., "feedback_points": [...]}]
        """
        for sample in feedback_samples:
            self.pool.append(sample)

    def sample(self, batch_size):
        """从缓冲区随机采样完整序列。"""
        if len(self.pool) == 0:
            return []
            
        actual_batch_size = min(batch_size, len(self.pool))
        return random.sample(self.pool, actual_batch_size)

    def __len__(self):
        return len(self.pool)

    def is_ready(self, batch_size):
        # 一旦我们至少有一个完整的批次，就可以开始训练
        return len(self.pool) >= batch_size
