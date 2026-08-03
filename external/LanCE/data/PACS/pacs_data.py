from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import sys
from tqdm import tqdm
import clip

# Makes `from utils import ...` resolve whether this module is imported as a
# package (data.PACS.pacs_data, external/LanCE already on sys.path via the
# main entry point's cwd) or run directly as a script (python data/PACS/
# pacs_data.py, where sys.path[0] would otherwise be data/PACS/ itself).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import get_domain_text_embs


class Processed_PACS_Dataset(Dataset):
    """One class handles all 4 PACS domains x both splits (unlike CUB's
    separate source-only/target-only classes) - every PACS domain needs
    both a train and a test split, there's no fixed source/target asymmetry.

    domain_diffs only depends on classname2id + the fixed 204-descriptor
    pool (prompts/prompt200new.py), not on any domain's actual images, so
    it should be computed once (pass src_dm_texts/tgt_dm_texts for exactly
    one domain instance) and reused for the other three (pass the resulting
    classname2id/concept2id in, and copy .domain_diffs onto the later
    instances - see data/__init__.py's get_pacs_datasets).
    """

    def __init__(self, args, data_root, domain, split, meta_root="data/PACS",
                 classname2id=None, concept2id=None,
                 src_dm_texts=None, tgt_dm_texts=None):
        if split not in ("train", "test"):
            raise ValueError("split must be train or test")

        with open(os.path.join(meta_root, f"pacs_{domain}_{split}.txt"), "r") as f:
            self.annos = f.readlines()

        self.domain = domain
        self.data_root = os.path.join(data_root, domain)

        device = args.device
        self.clip_model, self.preprocess = clip.load(args.CLIP_type, device=device)

        if classname2id is None:
            with open(os.path.join(meta_root, "pacs_classes.txt"), "r") as f:
                classname2id = {x.rstrip(): i for i, x in enumerate(f.readlines())}
        self.classname2id = classname2id

        if concept2id is None:
            with open(os.path.join(meta_root, "pacs_concepts.txt"), "r") as f:
                concept2id = {x.rstrip(): c_id for c_id, x in enumerate(f.readlines())}
        self.concept2id = concept2id

        self.domain_diffs = None
        if src_dm_texts is not None and tgt_dm_texts is not None:
            self.domain_diffs = []
            print(f"----------Computing Domain Differences ({domain})----------")
            for src_prompts, tgt_prompts in tqdm(zip(src_dm_texts * len(tgt_dm_texts), tgt_dm_texts)):
                tqdm.write(tgt_prompts + " - " + src_prompts)
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
        label = int(cls_label)  # already 0-indexed by prepare_pacs_dataset.py

        attr_label = torch.tensor([0] * len(self.concept2id))
        return image, label, attr_label


if __name__ == "__main__":
    # Smoke test - run after prepare_pacs_dataset.py, e.g.:
    #   python data/PACS/pacs_data.py --data_dir <path with photo/, art_painting/, ...>/..
    # (--data_dir should be the parent of the PACS/pacs_raw/kfold folder, matching
    # get_pacs_datasets' data_root convention, OR pass the domain-containing folder
    # directly and adjust data_root below - this smoke test assumes the latter for
    # simplicity, since it bypasses get_pacs_datasets entirely.)
    from args import get_args
    from prompts.prompt200new import source_text_prompts, target_text_prompts

    args = get_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ds = Processed_PACS_Dataset(
        args, data_root=args.data_dir, domain="photo", split="train",
        meta_root="data/PACS", src_dm_texts=source_text_prompts, tgt_dm_texts=target_text_prompts,
    )
    assert len(ds.classname2id) == 7, f"expected 7 classes, got {len(ds.classname2id)}"
    assert len(ds.concept2id) == 70, f"expected 70 concepts, got {len(ds.concept2id)}"
    assert ds.domain_diffs.shape[1:] == (7, 768), f"unexpected domain_diffs shape {ds.domain_diffs.shape}"
    image, label, attr_label = ds[0]
    assert image.shape == (3, 224, 224), f"unexpected image shape {image.shape}"
    assert 0 <= label < 7
    assert attr_label.shape == (70,)
    print(f"OK - {len(ds)} training images, classes={list(ds.classname2id.keys())}, "
          f"domain_diffs={tuple(ds.domain_diffs.shape)}")
