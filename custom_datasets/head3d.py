import cv2
import json
import random
import numpy as np

from torch.utils.data import Dataset


class Head3dFlameDataset(Dataset):
    def __init__(self, size=384):
        self.data = []
        with open('./data/head3d_flame/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.size = size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./data/head3d_flame/' + source_filename)
        if random.randint(0, 100):
            target = cv2.imread('./data/head3d_flame/' + target_filename)
        else:
            target = source.copy()
            prompt = 'worst quality, low quality, bald, close-up, portrait, error eyes, error mouth'

        rd = random.randint(0, 100)
        if rd == 0:
            lowsize = random.randint(32, 64)
            _source = cv2.resize(source, (lowsize,lowsize), interpolation=cv2.INTER_LINEAR)
            source = cv2.resize(_source, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_LINEAR)
        elif rd == 1:
            source = cv2.GaussianBlur(source, (5,5), 1)

        rd = random.randint(0, 100)
        if rd == 0:
            lowsize = random.randint(32, 64)
            _target = cv2.resize(target, (lowsize,lowsize), interpolation=cv2.INTER_LINEAR)
            target = cv2.resize(_target, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)
            prompt = 'lowres, ' + prompt
        elif rd == 1:
            target = cv2.GaussianBlur(target, (5,5), 1)
            prompt = 'blur, ' + prompt

        source = cv2.resize(source, (self.size,self.size), interpolation=cv2.INTER_LANCZOS4)
        target = cv2.resize(target, (self.size,self.size), interpolation=cv2.INTER_LANCZOS4)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1]..
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


class Head3dHifi3dDataset(Dataset):
    def __init__(self, size=384):
        self.data = []
        with open('./data/head3d_hifi3d/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.size = size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./data/head3d_hifi3d/' + source_filename)
        if random.randint(0, 100):
            target = cv2.imread('./data/head3d_hifi3d/' + target_filename)
        else:
            target = source.copy()
            prompt = 'worst quality, low quality, bald, close-up, portrait, error eyes, error mouth'

        rd = random.randint(0, 100)
        if rd:
            size = source.shape[0]
            new_size = random.randint(int(source.shape[0] / 3 * 2), source.shape[0] - 8)
            x0 = random.randint(0, size - new_size)
            y0 = random.randint(0, size - new_size)
            x1 = x0 + new_size
            y1 = y0 + new_size
            source = source[y0:y1,x0:x1]
            target = target[y0:y1,x0:x1]

        rd = random.randint(0, 100)
        if rd == 0:
            lowsize = random.randint(32, 64)
            _source = cv2.resize(source, (lowsize,lowsize), interpolation=cv2.INTER_LINEAR)
            source = cv2.resize(_source, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_LINEAR)
        elif rd == 1:
            source = cv2.GaussianBlur(source, (5,5), 1)

        rd = random.randint(0, 100)
        if rd == 0:
            lowsize = random.randint(32, 64)
            _target = cv2.resize(target, (lowsize,lowsize), interpolation=cv2.INTER_LINEAR)
            target = cv2.resize(_target, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)
            prompt = 'lowres, ' + prompt
        elif rd == 1:
            target = cv2.GaussianBlur(target, (5,5), 1)
            prompt = 'blur, ' + prompt

        source = cv2.resize(source, (self.size,self.size), interpolation=cv2.INTER_LANCZOS4)
        target = cv2.resize(target, (self.size,self.size), interpolation=cv2.INTER_LANCZOS4)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1]..
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


class Head3dPano3dDataset(Dataset):
    def __init__(self, size=384):
        self.data = []
        with open('./data/head3d_pano3d/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.size = size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./data/head3d_pano3d/' + source_filename)
        if random.randint(0, 100):
            target = cv2.imread('./data/head3d_pano3d/' + target_filename)
        else:
            target = source.copy()
            prompt = 'worst quality, low quality, bald, close-up, portrait, error eyes, error mouth'

        rd = random.randint(0, 1)
        if rd:
            size = source.shape[0]
            new_size = random.randint(int(source.shape[0] / 3 * 2), source.shape[0] - 8)
            x0 = random.randint(0, size - new_size)
            y0 = random.randint(0, size - new_size)
            x1 = x0 + new_size
            y1 = y0 + new_size
            source = source[y0:y1,x0:x1]
            target = target[y0:y1,x0:x1]

        rd = random.randint(0, 100)
        if rd == 0:
            lowsize = random.randint(32, 64)
            _source = cv2.resize(source, (lowsize,lowsize), interpolation=cv2.INTER_LINEAR)
            source = cv2.resize(_source, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_LINEAR)
        elif rd == 1:
            source = cv2.GaussianBlur(source, (5,5), 1)

        rd = random.randint(0, 100)
        if rd == 0:
            lowsize = random.randint(32, 64)
            _target = cv2.resize(target, (lowsize,lowsize), interpolation=cv2.INTER_LINEAR)
            target = cv2.resize(_target, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)
            prompt = 'lowres, ' + prompt
        elif rd == 1:
            target = cv2.GaussianBlur(target, (5,5), 1)
            prompt = 'blur, ' + prompt

        source = cv2.resize(source, (self.size,self.size), interpolation=cv2.INTER_LANCZOS4)
        target = cv2.resize(target, (self.size,self.size), interpolation=cv2.INTER_LANCZOS4)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1]..
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


class Head3dDataset(Dataset):
    def __init__(self, size=384):
        self.dataset_flame = Head3dFlameDataset(size)
        self.dataset_hifi3d = Head3dHifi3dDataset(size)
        self.dataset_pano3d = Head3dPano3dDataset(size)
        self.length = len(self.dataset_flame) + len(self.dataset_hifi3d) + len(self.dataset_pano3d)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < len(self.dataset_flame):
            return self.dataset_flame[idx]
        elif idx < len(self.dataset_flame) + len(self.dataset_hifi3d):
            return self.dataset_hifi3d[idx - len(self.dataset_flame)]
        else:
            return self.dataset_pano3d[idx - len(self.dataset_flame) - len(self.dataset_hifi3d)]
