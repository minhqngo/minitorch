import numpy as np
from .tensor.functions import tensor


class DataLoader:
    def __init__(self, dataset, backend, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.backend = backend
        
    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))
    
    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)
            
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_data = [self.dataset[j] for j in batch_indices]
            
            inputs, labels = zip(*batch_data)

            inputs_tensor = tensor(list(inputs), backend=self.backend)
            labels_tensor = tensor(list(labels), backend=self.backend)

            yield inputs_tensor, labels_tensor