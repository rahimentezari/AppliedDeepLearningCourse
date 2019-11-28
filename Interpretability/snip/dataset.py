import os
import itertools
import numpy as np
import cv2
import matplotlib.pyplot as plt
import mnist
import cifar


def twodnopad_threedpad(x):
    features_3d = np.zeros((x.shape[0], 32, 32, 3))
    for i in range(x.shape[0]):
        feature_pad = np.pad(x[i], ((2, 2), (2, 2), (0, 0)), 'constant')
        feature_3d_pad = cv2.cvtColor(feature_pad, cv2.COLOR_GRAY2RGB)
        features_3d[i] = feature_3d_pad
    return features_3d


class Dataset(object):
    def __init__(self, datasource, path_data, **kwargs):
        self.datasource = datasource
        self.path_data = path_data
        self.rand = np.random.RandomState(42)
        if self.datasource == 'mnist':
            self.num_classes = 10
            self.dataset = mnist.read_data(os.path.join(self.path_data, 'MNIST'))
            # print(self.dataset['train']['input'].shape)
            # print(self.dataset['train']['input'][0][:][:].shape)
            plt.imsave('mnist1.png',self.dataset['train']['input'][0][:][:].reshape(28, 28) )
            self.dataset['train']['input'] = twodnopad_threedpad(self.dataset['train']['input'])
            # self.dataset['val']['input'] = twodnopad_threedpad(self.dataset['val']['input'])
            self.dataset['test']['input'] = twodnopad_threedpad(self.dataset['test']['input'])
            print("self.dataset['train']['input'].shape", self.dataset['train']['input'].shape)
            # print(self.dataset['test']['input'][0:10])
        elif self.datasource == 'cifar-10':
            self.num_classes = 10
            self.dataset = cifar.read_data(os.path.join(self.path_data, 'cifar-10-batches-py'))
        else:
            raise NotImplementedError
        self.split_dataset('train', 'val', int(self.dataset['train']['input'].shape[0] * 0.1),
            self.rand)
        self.num_example = {k: self.dataset[k]['input'].shape[0] for k in self.dataset.keys()}
        self.example_generator = {
            'train': self.iterate_example('train'),
            'val': self.iterate_example('val'),
            'test': self.iterate_example('test', shuffle=False),
        }

    def iterate_example(self, mode, shuffle=True):
        epochs = itertools.count()
        for i in epochs:
            example_ids = list(range(self.num_example[mode]))
            if shuffle:
                self.rand.shuffle(example_ids)
            for example_id in example_ids:
                yield {
                    'input': self.dataset[mode]['input'][example_id],
                    'label': self.dataset[mode]['label'][example_id],
                    'id': example_id,
                }

    def get_next_batch(self, mode, batch_size):
        inputs, labels, ids = [], [], []
        for i in range(batch_size):
            example = next(self.example_generator[mode])
            inputs.append(example['input'])
            labels.append(example['label'])
            ids.append(example['id'])
        return {
            'input': np.asarray(inputs),
            'label': np.asarray(labels),
            'id': np.asarray(ids),
        }

    def generate_example_epoch(self, mode):
        example_ids = range(self.num_example[mode])
        for example_id in example_ids:
            yield {
                'input': self.dataset[mode]['input'][example_id],
                'label': self.dataset[mode]['label'][example_id],
                'id': example_id,
            }

    def generate_sample(self, mode):
        example_ids = range(100)
        for example_id in example_ids:
            yield {
                'input': self.dataset[mode]['input'][example_id],
                'label': self.dataset[mode]['label'][example_id],
                'id': example_id,
            }

    def split_dataset(self, source, target, number, rand):
        keys = ['input', 'label']
        indices = list(range(self.dataset[source]['input'].shape[0]))
        rand.shuffle(indices)
        ind_target = indices[:number]
        ind_remain = indices[number:]
        self.dataset[target] = {k: self.dataset[source][k][ind_target] for k in keys}
        self.dataset[source] = {k: self.dataset[source][k][ind_remain] for k in keys}









