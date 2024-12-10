from os.path import join as pjoin

import numpy as np
from .humanml.utils.word_vectorizer import WordVectorizer
from .HumanML3D import HumanML3DDataModule
from .Kit import KitDataModule
from .Humanact12 import Humanact12DataModule
from .Uestc import UestcDataModule
from .utils import *


def get_mean_std(phase, cfg, dataset_name):
    # if phase == 'gt':
    #     # used by T2M models (including evaluators)
    #     mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    #     std = np.load(pjoin(opt.meta_dir, 'std.npy'))
    # elif phase in ['train', 'val', 'text_only']:
    #     # used by our models
    #     mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    #     std = np.load(pjoin(opt.data_root, 'Std.npy'))

    # todo: use different mean and val for phases
    name = "t2m" if dataset_name == "humanml3d" else dataset_name
    assert name in ["t2m", "kit"]
    # if phase in ["train", "val", "test"]:
    if phase in ["val"]:
        if name == 't2m':
            data_root = pjoin(cfg.model.t2m_path, name, "Comp_v6_KLD01",
                              "meta")
        elif name == 'kit':
            data_root = pjoin(cfg.model.t2m_path, name, "Comp_v6_KLD005",
                              "meta")
        else:
            raise ValueError("Only support t2m and kit")
        # 아마 mean.npy, std.npy 와 아래 Mean.npy, Std.npy는 HumanML3D 데이터셋을 가져오면 생성되는것 같다.
        mean = np.load(pjoin(data_root, "mean.npy"))
        std = np.load(pjoin(data_root, "std.npy"))
    else:
        data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
        mean = np.load(pjoin(data_root, "Mean.npy"))
        std = np.load(pjoin(data_root, "Std.npy"))

    return mean, std


def get_WordVectorizer(cfg, phase, dataset_name):
    if phase not in ["text_only"]:
        if dataset_name.lower() in ["humanml3d", "kit"]:
            return WordVectorizer(cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")
        else:
            raise ValueError("Only support WordVectorizer for HumanML3D")
    else:
        return None


def get_collate_fn(name, phase="train"):
    if name.lower() in ["humanml3d", "kit"]:
        return mld_collate
    elif name.lower() in ["humanact12", 'uestc']:
        return a2m_collate
    # else:
    #     return all_collate
    # if phase == "test":
    #     return eval_collate
    # else:


# map config name to module&path
dataset_module_map = {
    "humanml3d": HumanML3DDataModule,
    "kit": KitDataModule,
    "humanact12": Humanact12DataModule,
    "uestc": UestcDataModule,
}
motion_subdir = {"humanml3d": "new_joint_vecs", "kit": "new_joint_vecs"}


def get_datasets(cfg, logger=None, phase="train"):
    # get dataset names form cfg
    dataset_names = eval(f"cfg.{phase.upper()}.DATASETS")                   # humanml3d
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name.lower() in ["humanml3d", "kit"]:                    # humanml3d default
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")    # cfg.DATASET.HUMANML3D.ROOT 반환, './datasets/humanml3d'
            # get mean and std corresponding to dataset
            mean, std = get_mean_std(phase, cfg, dataset_name)              # dataset의 mean, std값 load.
            mean_eval, std_eval = get_mean_std("val", cfg, dataset_name)
            # get WordVectorizer
            wordVectorizer = get_WordVectorizer(cfg, phase, dataset_name)
            # get collect_fn
            collate_fn = get_collate_fn(dataset_name, phase)
            # get dataset module
            dataset = dataset_module_map[dataset_name.lower()](
                cfg=cfg,
                batch_size=cfg.TRAIN.BATCH_SIZE,        # 왜 1이 아니고, TRAIN_BATCH_SIZE를 썼을 까?
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                collate_fn=collate_fn,
                mean=mean,
                std=std,
                mean_eval=mean_eval,
                std_eval=std_eval,
                w_vectorizer=wordVectorizer,
                text_dir=pjoin(data_root, "texts"),
                motion_dir=pjoin(data_root, motion_subdir[dataset_name]),   # data_root: './datasets/humanml3d'
                                                                            # motion_subdir: 'new_joint_vecs'
                max_motion_length=cfg.DATASET.SAMPLER.MAX_LEN,
                min_motion_length=cfg.DATASET.SAMPLER.MIN_LEN,
                max_text_len=cfg.DATASET.SAMPLER.MAX_TEXT_LEN,
                unit_length=eval(
                    f"cfg.DATASET.{dataset_name.upper()}.UNIT_LEN"),
            )
            datasets.append(dataset)
        elif dataset_name.lower() in ["humanact12", 'uestc']:
            # get collect_fn
            collate_fn = get_collate_fn(dataset_name, phase)
            # get dataset module
            dataset = dataset_module_map[dataset_name.lower()](
                datapath=eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT"),
                cfg=cfg,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                collate_fn=collate_fn,
                num_frames=cfg.DATASET.HUMANACT12.NUM_FRAMES,
                sampling=cfg.DATASET.SAMPLER.SAMPLING,
                sampling_step=cfg.DATASET.SAMPLER.SAMPLING_STEP,
                pose_rep=cfg.DATASET.HUMANACT12.POSE_REP,
                max_len=cfg.DATASET.SAMPLER.MAX_LEN,
                min_len=cfg.DATASET.SAMPLER.MIN_LEN,
                num_seq_max=cfg.DATASET.SAMPLER.MAX_SQE
                if not cfg.DEBUG else 100,
                glob=cfg.DATASET.HUMANACT12.GLOB,
                translation=cfg.DATASET.HUMANACT12.TRANSLATION)
            cfg.DATASET.NCLASSES = dataset.nclasses
            datasets.append(dataset)
        elif dataset_name.lower() in ["amass"]:
            # todo: add amass dataset
            raise NotImplementedError
        else:
            raise NotImplementedError
    cfg.DATASET.NFEATS = datasets[0].nfeats         # sample에서 뽑아옴
    cfg.DATASET.NJOINTS = datasets[0].njoints       # 22
    return datasets
