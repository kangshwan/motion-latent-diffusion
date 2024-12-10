import numpy as np
import torch

from mld.data.humanml.scripts.motion_process import (process_file,
                                                     recover_from_ric)

from .base import BASEDataModule
from .humanml.data.dataset import Text2MotionDatasetV2, TextOnlyDataset


class HumanML3DDataModule(BASEDataModule):

    def __init__(self,
                 cfg,
                 batch_size,
                 num_workers,
                 collate_fn=None,
                 phase="train",
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=collate_fn)
        self.save_hyperparameters(logger=False) # self.hparam 만들어짐.
                                                # __init__에 전달된 인자를 자동으로 hparam에 저장.
        self.name = "humanml3d"
        self.njoints = 22
        if phase == "text_only":
            self.Dataset = TextOnlyDataset
        else:
            # 여기서 class를 다
            self.Dataset = Text2MotionDatasetV2
        self.cfg = cfg
        sample_overrides = {
            "split": "val",
            "tiny": True,
            "progress_bar": False
        }
        self._sample_set = self.get_sample_set(overrides=sample_overrides)
        # Get additional info of the dataset
        self.nfeats = self._sample_set.nfeats
        # self.transforms = self._sample_set.transforms

    def feats2joints(self, features):
        # features가 vae를 통과했기 때문에, 평균과 분산을 곱해 원래 데이터 공간으로 매핑하는 과정이다!
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, self.njoints)

    def joints2feats(self, features):
        features = process_file(features, self.njoints)[0]
        # mean = torch.tensor(self.hparams.mean).to(features)
        # std = torch.tensor(self.hparams.std).to(features)
        # features = (features - mean) / std
        return features

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        # random select samples for mm
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.TEST.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
