from spektral.data import Dataset
from spektral.data import Loader

class Twitch(Dataset):
    def __init__(self, transforms=None, **kwargs):
        super(Twitch, self).__init__(transforms, **kwargs)

    def read(self):
        pass