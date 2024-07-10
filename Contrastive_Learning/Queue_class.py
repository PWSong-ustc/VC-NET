import torch
import torch.nn as nn


class Queue():
    """
    Build a query
    """

    def __init__(self, dim=2048, K=65536, device=0):
        """
        dim: feature dimension (default: 2048)
        K: queue size; number of negative keys (default: 65536)
        """
        self.iter = 0
        self.K = K

        # create the queue
        self.queue = torch.zeros(45, 45, dim, K)
        self.queue = nn.functional.normalize(self.queue, dim=0).cuda(device)

        self.label = torch.zeros(720, 720, K)   # crop后的大小
        self.label = nn.functional.normalize(self.label, dim=0).cuda(device)

        self.feature_queue_ptr = torch.zeros(1, dtype=torch.long).cuda(device)
        self.label_queue_ptr = torch.zeros(1, dtype=torch.long).cuda(device)

    # feature queue
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue

        batch_size = keys.shape[0]

        ptr = int(self.feature_queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, :, :, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.feature_queue_ptr[0] = ptr
        self.iter += 1 
    
    # label queue
    def _dequeue_and_enqueue_label(self, keys):
        # gather keys before updating queue

        batch_size = keys.shape[0]

        ptr = int(self.label_queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.label[:, :, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.label_queue_ptr[0] = ptr

