import glob
import re

import os.path as osp

from .bases import BaseImageDataset


class VeRi(BaseImageDataset):
    """
       VeRi-776
       Reference:
       Liu, Xinchen, et al. "Large-scale vehicle re-identification in urban surveillance videos." ICME 2016.

       URL:https://vehiclereid.github.io/VeRi/

       Dataset statistics:
       # identities: 776
       # images: 37778 (train) + 1678 (query) + 11579 (gallery)
       # cameras: 20
       """

    dataset_dir = 'veri'

    def __init__(self, root='query', verbose=True, **kwargs):
        super(VeRi, self).__init__()
        self.query_dir = root  # 'query'
        query = self._process_dir(self.query_dir, relabel=False)


        if verbose:
            self.print_dataset_statistics(query)
        self.query = query

        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)


    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 776  # pid == 0 means background
            assert 1 <= camid <= 20
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

