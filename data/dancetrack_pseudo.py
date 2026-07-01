# Copyright (c) Ruopeng Gao. All Rights Reserved.
# Modified for PS-MOT (ECCV 2026), CVPU Kai Luo. All Rights Reserved.

import os
import torch
from collections import defaultdict
from configparser import ConfigParser

from .one_dataset import OneDataset
from .util import is_legal, append_annotation


class DanceTrackPseudo(OneDataset):
    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "DanceTrack",
            split: str = "train",
            load_annotation: bool = True,
            pseudo_label_file: str = "sam2_pgt.txt",
    ):
        super().__init__(
            data_root=data_root,
            sub_dir=sub_dir,
            split=split,
            load_annotation=load_annotation,
        )
        self.pseudo_label_file = pseudo_label_file

        self.sequence_infos = self._get_sequence_infos()
        self.image_paths = self._get_image_paths()
        if self.load_annotation:
            self.annotations = self._get_annotations()
        return

    def _get_sequence_names(self):
        return os.listdir(os.path.join(self.data_dir, self.split))

    def _get_sequence_infos(self):
        sequence_names = self._get_sequence_names()
        sequence_infos = dict()
        for sequence_name in sequence_names:
            sequence_dir = self._get_sequence_dir(self.data_dir, self.split, sequence_name)
            ini = ConfigParser()
            ini.read(os.path.join(sequence_dir, "seqinfo.ini"))
            sequence_infos[sequence_name] = {
                "width": int(ini["Sequence"]["imWidth"]),
                "height": int(ini["Sequence"]["imHeight"]),
                "length": int(ini["Sequence"]["seqLength"]),
                "is_static": False,
            }
        return sequence_infos

    def _get_image_paths(self):
        sequence_names = self._get_sequence_names()
        image_paths = defaultdict(list)
        for sequence_name in sequence_names:
            sequence_dir = self._get_sequence_dir(self.data_dir, self.split, sequence_name)
            for i in range(self.sequence_infos[sequence_name]["length"]):
                image_paths[sequence_name].append(self._get_image_path(sequence_dir, i))
        return image_paths

    @staticmethod
    def _get_sequence_dir(data_dir, split, sequence_name):
        return str(os.path.join(data_dir, split, sequence_name))

    @staticmethod
    def _get_image_path(sequence_dir, frame_idx):
        return str(os.path.join(sequence_dir, "img1", f"{frame_idx+1:08d}.jpg"))

    def _get_annotations(self):
        sequence_names = self._get_sequence_names()
        annotations = self._init_annotations(sequence_names)
        for sequence_name in sequence_names:
            sequence_dir = self._get_sequence_dir(self.data_dir, self.split, sequence_name)
            gt_file_path = os.path.join(sequence_dir, "gt", self.pseudo_label_file)
            if not os.path.exists(gt_file_path):
                gt_file_path = os.path.join(sequence_dir, "gt", "gt.txt")
            is_pseudo_label_file = os.path.basename(gt_file_path) == self.pseudo_label_file
            with open(gt_file_path, "r") as gt_file:
                for line in gt_file:
                    line = line.strip().split(",")
                    frame_id, obj_id, x, y, w, h, *extra_fields = line
                    frame_id, obj_id = map(int, [frame_id, obj_id])
                    x, y, w, h = map(float, [x, y, w, h])
                    quality = float(extra_fields[-1]) if is_pseudo_label_file and len(extra_fields) >= 1 else 1.0
                    bbox = [x, y, w, h]
                    category, visibility = 0, 1.0
                    ann_index = frame_id - 1
                    annotations[sequence_name][ann_index] = append_annotation(
                        annotation=annotations[sequence_name][ann_index],
                        obj_id=obj_id,
                        category=category,
                        bbox=bbox,
                        visibility=visibility,
                        quality=quality,
                    )
        for sequence_name in sequence_names:
            for i in range(self.sequence_infos[sequence_name]["length"]):
                annotations[sequence_name][i]["is_legal"] = is_legal(annotations[sequence_name][i])
        return annotations

    def _init_annotations(self, sequence_names):
        annotations = dict()
        for sequence_name in sequence_names:
            annotations[sequence_name] = []
            for _ in range(self.sequence_infos[sequence_name]["length"]):
                annotations[sequence_name].append({
                    "id": torch.zeros((0, ), dtype=torch.int64),
                    "category": torch.zeros((0, ), dtype=torch.int64),
                    "bbox": torch.zeros((0, 4), dtype=torch.float32),
                    "visibility": torch.zeros((0, ), dtype=torch.float32),
                    "quality": torch.zeros((0, ), dtype=torch.float32),
                })
        return annotations


DanceTrackS = DanceTrackPseudo
