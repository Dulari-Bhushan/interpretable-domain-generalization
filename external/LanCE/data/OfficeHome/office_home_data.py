from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import sys
from tqdm import tqdm
import clip

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import get_domain_text_embs

# domain key -> actual on-disk folder name (mirrors prepare_office_home_dataset.py)
DOMAIN_FOLDERS = {
    "art": "Art",
    "clipart": "Clipart",
    "product": "Product",
    "real_world": "Real World",
}


class Processed_OfficeHome_Dataset(Dataset):
    """Mirrors Processed_PACS_Dataset's interface/design exactly (see
    data/PACS/pacs_data.py for the full rationale) - one class handles all
    4 Office-Home domains x both splits, sharing one classname2id/concept2id/
    domain_diffs across domains (computed once, passed into later instances)."""

    def __init__(self, args, data_root, domain, split, meta_root="data/OfficeHome",
                 classname2id=None, concept2id=None,
                 src_dm_texts=None, tgt_dm_texts=None):
        if split not in ("train", "test"):
            raise ValueError("split must be train or test")

        with open(os.path.join(meta_root, f"office_home_{domain}_{split}.txt"), "r") as f:
            self.annos = f.readlines()

        self.domain = domain
        self.data_root = os.path.join(data_root, DOMAIN_FOLDERS[domain])

        device = args.device
        self.clip_model, self.preprocess = clip.load(args.CLIP_type, device=device)

        if classname2id is None:
            with open(os.path.join(meta_root, "office_home_classes.txt"), "r") as f:
                classname2id = {x.rstrip(): i for i, x in enumerate(f.readlines())}
        self.classname2id = classname2id

        if concept2id is None:
            with open(os.path.join(meta_root, "office_home_concepts.txt"), "r") as f:
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
        label = int(cls_label)  # already 0-indexed by prepare_office_home_dataset.py

        attr_label = torch.tensor([0] * len(self.concept2id))
        return image, label, attr_label


if __name__ == "__main__":
    # Smoke test - run after prepare_office_home_dataset.py, e.g.:
    #   python data/OfficeHome/office_home_data.py --data_dir <path with Art/, Clipart/, ...>
    from args import get_args
    from prompts.prompt200new import source_text_prompts, target_text_prompts

    args = get_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ds = Processed_OfficeHome_Dataset(
        args, data_root=args.data_dir, domain="art", split="train",
        meta_root="data/OfficeHome", src_dm_texts=source_text_prompts, tgt_dm_texts=target_text_prompts,
    )
    # office_home_concepts.txt has 260 written lines, 260 written phrases,
    # 65 classes x 4/class - but concept2id is a dict keyed by phrase text,
    # so 3 phrases reused verbatim across different classes ("a flat base"
    # for bottle/mug, "a long narrow handle" for fork/spoon, "four
    # supporting legs" for bed/table) collapse to one entry each. That's a
    # feature, not a bug, for a concept-bottleneck model - shared attributes
    # across classes is the whole point (CUB's own concept bank works the
    # same way) - so the unique count is 257, not 260.
    n_concepts = len(ds.concept2id)
    assert len(ds.classname2id) == 65, f"expected 65 classes, got {len(ds.classname2id)}"
    assert n_concepts == 257, f"expected 257 unique concepts, got {n_concepts}"
    assert ds.domain_diffs.shape[1:] == (65, 768), f"unexpected domain_diffs shape {ds.domain_diffs.shape}"
    image, label, attr_label = ds[0]
    assert image.shape == (3, 224, 224), f"unexpected image shape {image.shape}"
    assert 0 <= label < 65
    assert attr_label.shape == (n_concepts,)
    print(f"OK - {len(ds)} training images, {len(ds.classname2id)} classes, "
          f"domain_diffs={tuple(ds.domain_diffs.shape)}")
