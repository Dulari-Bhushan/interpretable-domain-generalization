import os
from .CUB.cub_data import Processed_CUB_Dataset, Processed_CUBP_Dataset
from .awa2.awa2_data import Processed_awa2, Processed_awa2p
from .RIVAL10.rival10_data import CBM_RIVAL10, CBM_RIVAL10S
from .LAD_animal.lad_A_data import Processed_LADA,Processed_LADAS
from .LAD_vehicle.lad_V_data import Processed_LADV,Processed_LADV3
from .PACS.pacs_data import Processed_PACS_Dataset
from .OfficeHome.office_home_data import Processed_OfficeHome_Dataset

from prompts.prompt200new import source_text_prompts, target_text_prompts
import torch
from torch.utils.data import DataLoader

PACS_DOMAINS = ("photo", "art_painting", "cartoon", "sketch")
OFFICE_HOME_DOMAINS = ("art", "clipart", "product", "real_world")

def get_dataset_classes(args):
    if args.dataset == "CUB":
        # source train
        train_dataset = Processed_CUB_Dataset(args,
                                              data_root=os.path.join(args.data_dir,"CUB/CUB_200_2011/images"),
                                              split="train",
                                              meta_root="data/CUB",
                                              attr_name="CUBpath2attr.pkl",
                                              src_dm_texts = source_text_prompts, tgt_dm_texts = target_text_prompts)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        # source test
        test_dataset = Processed_CUB_Dataset(args,
                                             data_root=os.path.join(args.data_dir,"CUB/CUB_200_2011/images"),
                                             split="test",
                                             meta_root="data/CUB",
                                             attr_name="CUBpath2attr.pkl")
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        # target test
        target_test_dataset = Processed_CUBP_Dataset(args,
                                                     data_root=os.path.join(args.data_dir,"CUB/CUB-200-Painting"),
                                                     split="test",
                                                     meta_root="data/CUB",
                                                     classname2id=train_dataset.classname2id,concept2id=train_dataset.concept2id)
        target_test_loader = DataLoader(target_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)


    elif args.dataset == "RIVAL10":
        train_dataset = CBM_RIVAL10(args,data_root=os.path.join(args.data_dir,"/RIVAL10"), split="train",meta_root="data/RIVAL10",
                                    src_dm_texts = source_text_prompts, tgt_dm_texts = target_text_prompts)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_dataset = CBM_RIVAL10(args,data_root=os.path.join(args.data_dir,"RIVAL10"), split="test",meta_root="data/RIVAL10")
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        target_test_dataset = CBM_RIVAL10S(args,data_root=os.path.join(args.data_dir,"RIVAL10-Sketch"), split="test",
                                           classname2id=train_dataset.classname2id,concept2id=train_dataset.classname2id)
        target_test_loader = DataLoader(target_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    elif args.dataset == "AWA2":
        train_dataset = Processed_awa2(args,
                                      data_root=os.path.join(args.data_dir,"/AwA2/JPEGImages"),
                                      split="train",
                                      meta_root="data/awa2",
                                      attr_name = None,
                                      src_dm_texts = source_text_prompts, tgt_dm_texts = target_text_prompts)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        test_dataset = Processed_awa2(args,
                                     data_root=os.path.join(args.data_dir,"AwA2/JPEGImages"),
                                     split="test",
                                     meta_root="data/awa2",
                                     attr_name= None)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        target_test_dataset = Processed_awa2p(args,
                                             data_root=os.path.join(args.data_dir,"AwA2_clipart"),
                                             split="test",
                                             meta_root="data/awa2",
                                             classname2id=train_dataset.classname2id,concept2id=train_dataset.classname2id)
        target_test_loader = DataLoader(target_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)


    elif args.dataset == "LADA":
        train_dataset = Processed_LADA(args,
                                      data_root=os.path.join(args.data_dir,"LAD_animal/animals"),
                                      split="train",
                                      meta_root="data/LAD_animal",
                                      attr_name = None,
                                      src_dm_texts = source_text_prompts, tgt_dm_texts = target_text_prompts)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        test_dataset = Processed_LADA(args,
                                     data_root=os.path.join(args.data_dir,"LAD_animal/animals"),
                                     split="test",
                                     meta_root="data/LAD_animal",
                                     attr_name= None)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        target_test_dataset = Processed_LADAS(args,
                                             data_root=os.path.join(args.data_dir,"LAD_animal/sculpture-animals"),
                                             split="test",
                                             meta_root="data/LAD_animal",
                                             classname2id=train_dataset.classname2id,concept2id=train_dataset.classname2id)
        target_test_loader = DataLoader(target_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    elif args.dataset == "LADV":
        train_dataset = Processed_LADV(args,
                                       data_root=os.path.join(args.data_dir,"LAD_vehicle/vehicles"),
                                       split="train",
                                       meta_root="data/LAD_vehicle",
                                       attr_name=None,
                                       src_dm_texts=source_text_prompts, tgt_dm_texts=target_text_prompts)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        test_dataset = Processed_LADV(args,
                                      data_root=os.path.join(args.data_dir,"LAD_vehicle/vehicles"),
                                      split="test",
                                      meta_root="data/LAD_vehicle",
                                      attr_name=None)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        target_test_dataset = Processed_LADV3(args,
                                              data_root=os.path.join(args.data_dir,"LAD_vehicle/CAD-vehicles"),
                                              split="test",
                                              meta_root="data/LAD_vehicle",
                                              classname2id=train_dataset.classname2id,
                                              concept2id=train_dataset.classname2id)
        target_test_loader = DataLoader(target_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    else:
        raise NotImplementedError
    torch.cuda.empty_cache()
    return train_dataset, train_loader, test_dataset, test_loader, target_test_dataset, target_test_loader


def get_pacs_datasets(args, domains=PACS_DOMAINS):
    """PACS has 4 usable-as-either-train-or-eval domains, not one fixed
    source + one fixed target - get_dataset_classes' return signature can't
    express that, so this is a separate entry point rather than a new
    branch there. Used by experiments/domain_il.py's Domain-IL sweep.

    Returns:
        datasets: dict[domain] -> {"train": Dataset, "test": Dataset}
        classname2id, concept2id: dict[str, int], shared across all 4 domains
        domain_diffs: torch.Tensor, shared across all 4 domains (computed
            once from the first domain built; it depends only on
            classname2id + the fixed descriptor pool, not on any domain's
            actual images)
    """
    data_root = os.path.join(args.data_dir, "PACS", "pacs_raw", "kfold")
    datasets = {}
    classname2id = concept2id = domain_diffs = None
    for i, domain in enumerate(domains):
        train_dataset = Processed_PACS_Dataset(
            args, data_root=data_root, domain=domain, split="train",
            meta_root="data/PACS", classname2id=classname2id, concept2id=concept2id,
            src_dm_texts=source_text_prompts if i == 0 else None,
            tgt_dm_texts=target_text_prompts if i == 0 else None,
        )
        if i == 0:
            classname2id = train_dataset.classname2id
            concept2id = train_dataset.concept2id
            domain_diffs = train_dataset.domain_diffs
        else:
            train_dataset.domain_diffs = domain_diffs  # share, don't recompute
        test_dataset = Processed_PACS_Dataset(
            args, data_root=data_root, domain=domain, split="test",
            meta_root="data/PACS", classname2id=classname2id, concept2id=concept2id,
        )
        datasets[domain] = {"train": train_dataset, "test": test_dataset}
    torch.cuda.empty_cache()
    return datasets, classname2id, concept2id, domain_diffs


def get_office_home_datasets(args, domains=OFFICE_HOME_DOMAINS):
    """Same structure/rationale as get_pacs_datasets - see there for the
    full docstring. Office-Home: 4 domains (art/clipart/product/real_world),
    65 classes, used by Phase D (experiments/domain_il_officehome.py)."""
    data_root = os.path.join(args.data_dir, "OfficeHome", "office_home_raw")
    datasets = {}
    classname2id = concept2id = domain_diffs = None
    for i, domain in enumerate(domains):
        train_dataset = Processed_OfficeHome_Dataset(
            args, data_root=data_root, domain=domain, split="train",
            meta_root="data/OfficeHome", classname2id=classname2id, concept2id=concept2id,
            src_dm_texts=source_text_prompts if i == 0 else None,
            tgt_dm_texts=target_text_prompts if i == 0 else None,
        )
        if i == 0:
            classname2id = train_dataset.classname2id
            concept2id = train_dataset.concept2id
            domain_diffs = train_dataset.domain_diffs
        else:
            train_dataset.domain_diffs = domain_diffs
        test_dataset = Processed_OfficeHome_Dataset(
            args, data_root=data_root, domain=domain, split="test",
            meta_root="data/OfficeHome", classname2id=classname2id, concept2id=concept2id,
        )
        datasets[domain] = {"train": train_dataset, "test": test_dataset}
    torch.cuda.empty_cache()
    return datasets, classname2id, concept2id, domain_diffs