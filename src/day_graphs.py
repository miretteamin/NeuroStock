from torch_geometric.data  import InMemoryDataset
import torch

class DayGraphsCreation(InMemoryDataset):
    def __init__(self, root, data_list, transform=None):
        self.data_list = data_list
        #print(len(self.data_list))
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        #self.process()

    @property
    def processed_file_names(self):
        return 'data.pt'
    
    def process(self):
        torch.save(self.collate(self.data_list), self.processed_paths[0])


class DayGraphs(InMemoryDataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'data.pt'