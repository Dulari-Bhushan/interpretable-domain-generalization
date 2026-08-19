from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import sys
from tqdm import tqdm
import clip

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import get_domain_text_embs


class Processed_DomainNet_Dataset(Dataset):
    """Mirrors Processed_PACS_Dataset exactly (see data/PACS/pacs_data.py for
    the full rationale) - one class handles all 6 DomainNet domains x both
    splits. domain_diffs is computed once (first domain built) and shared,
    same convention as PACS/Office-Home.
    """

    def __init__(self, args, data_root, domain, split, meta_root="data/DomainNet",
                 classname2id=None, concept2id=None,
                 src_dm_texts=None, tgt_dm_texts=None):
        if split not in ("train", "test"):
            raise ValueError("split must be train or test")

        with open(os.path.join(meta_root, f"domainnet_{domain}_{split}.txt"), "r") as f:
            self.annos = f.readlines()

        self.domain = domain
        self.data_root = data_root  # paths in the manifest already start with "<domain>/..."

        device = args.device
        self.clip_model, self.preprocess = clip.load(args.CLIP_type, device=device)

        if classname2id is None:
            with open(os.path.join(meta_root, "domainnet_classes.txt"), "r") as f:
                classname2id = {x.rstrip(): i for i, x in enumerate(f.readlines())}
        self.classname2id = classname2id

        if concept2id is None:
            with open(os.path.join(meta_root, "domainnet_concepts.txt"), "r") as f:
                concept2id = {x.rstrip(): c_id for c_id, x in enumerate(f.readlines())}
        self.concept2id = concept2id

        self.domain_diffs = None
        if src_dm_texts is not None and tgt_dm_texts is not None:
            self.domain_diffs = []
            print(f"----------Computing Domain Differences ({domain})----------")
            for src_prompts, tgt_prompts in tqdm(zip(src_dm_texts * len(tgt_dm_texts), tgt_dm_texts)):
                source_embeddings, target_embeddings = get_domain_text_embs(
                    self.clip_model, [src_prompts], [tgt_prompts],
                    list(self.classname2id.keys()), device)
                source_embeddings /= source_embeddings.norm(dim=-1, keepdim=True)
                target_embeddings /= target_embeddings.norm(dim=-1, keepdim=True)
                diffs = target_embeddings.float() - source_embeddings.float()
                diffs /= diffs.norm(dim=-1, keepdim=True)
                self.domain_diffs.append(diffs)
            self.domain_diffs = torch.stack(self.domain_diffs, dim=0).to(device)
        self.clip_model = None

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        img_path, cls_label = self.annos[idx].strip().split(",")
        image = Image.open(os.path.join(self.data_root, img_path)).convert("RGB")
        image = self.preprocess(image)
        label = int(cls_label)

        attr_label = torch.tensor([0] * len(self.concept2id))
        return image, label, attr_label
