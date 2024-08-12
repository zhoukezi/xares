import copy
import json
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
from webdataset import TarWriter, WebLoader

from xares.audiowebdataset import create_rawaudio_webdataset, write_audio_tar
from xares.models import Mlp
from xares.task_base import TaskBase
from xares.utils import download_file, mkdir_if_not_exists, unzip_file


@dataclass
class ESC50Task(TaskBase):
    folds = range(1, 6)  # This dataset requires 5-fold validation in evaluation
    save_encoded_per_batches = 1000
    batch_size = 32
    trim_length = 220_500
    output_dim = 50
    metric = "accuracy"

    def __post_init__(self):
        self.ori_data_root = self.env_dir / "ESC-50-master"
        self.wds_audio_paths_dict = {fold: self.env_dir / f"wds-audio-fold-{fold}-*.tar" for fold in self.folds}
        self.wds_encoded_paths_dict = {fold: self.env_dir / f"wds-encoded-fold-{fold}-*.tar" for fold in self.folds}
        self.model = Mlp(in_features=self.encoder.output_dim, out_features=self.output_dim).to(self.encoder.device)
        self.checkpoint_dir = self.env_dir / "checkpoints"

        self.wds_encoded_training_fold_k = {
            fold: [f"{self.env_dir}/wds-encoded-fold-{f}-*.tar" for f in self.folds if f != fold] for fold in self.folds
        }

    def make_audio_tar(self):
        if not self.force_generate_audio_tar and self.audio_tar_ready_file.exists():
            logger.info(f"Skip making audio tar. {self.audio_tar_ready_file} already exists.")
            return

        # Download and extract ESC-50 dataset
        mkdir_if_not_exists(self.env_dir)
        download_file(
            "https://github.com/karoldvl/ESC-50/archive/master.zip",
            self.env_dir / "master.zip",
            force=self.force_download,
        )
        if not self.ori_data_root.exists():
            logger.info(f"Extracting {self.env_dir / 'master.zip'} to {self.env_dir}...")
            unzip_file(self.env_dir / "master.zip", self.env_dir)
        else:
            logger.info(f"Directory {self.ori_data_root} already exists. Skip.")

        # Create tar file with audio files
        df = pd.read_csv(self.ori_data_root / "meta/esc50.csv", usecols=["filename", "fold", "target"])
        df.filename = df.filename.apply(lambda x: (self.ori_data_root / "audio" / x).as_posix())

        assert df.fold.unique().tolist() == list(self.folds)
        for fold in self.folds:
            wds_audio_path = self.wds_audio_paths_dict[fold]
            df_split = df[df.fold == fold].drop(columns=["fold"])
            write_audio_tar(
                df_split.filename.tolist(),
                df_split.target.tolist(),
                wds_audio_path.as_posix(),
                force=self.force_generate_audio_tar,
            )

        self.audio_tar_ready_file.touch()

    def make_encoded_tar(self, num_shards: int = 20):
        def write_encoded_batches_to_wds(encoded_batches: List, ostream: TarWriter):

            for batch, label, keys in encoded_batches:
                for example, label, key in zip(batch, label, keys):
                    sample = {
                        "pth": example,
                        "json": json.dumps({"target": label["label"]}).encode("utf-8"),
                        "__key__": key,
                    }
                    ostream.write(sample)

        for fold in self.folds:
            if not self.force_generate_encoded_tar and self.encoded_tar_ready_file.exists():
                logger.info(f"Skip making encoded tar. {self.encoded_tar_ready_file} already exists.")
                continue

            logger.info(f"Encoding audio for fold {fold} ...")
            for shard in range(num_shards):
                sharded_tar_path = self.wds_audio_paths_dict[fold].as_posix().replace("*", f"0{shard:05d}")
                dl = create_rawaudio_webdataset(
                    [sharded_tar_path],
                    batch_size=self.batch_size,
                    num_workers=self.num_encoder_workers,
                    crop_size=self.trim_length,
                    with_json=True,
                )

                batch_buf = []
                sharded_encoded_tar_path = self.wds_encoded_paths_dict[fold].as_posix().replace("*", f"0{shard:05d}")
                with TarWriter(sharded_encoded_tar_path) as ostream:
                    for batch, length, label, keys in dl:
                        encoded_batch = self.encoder(batch, 44_100)
                        batch_buf.append([encoded_batch, label, keys])

                        if len(batch_buf) >= self.save_encoded_per_batches:
                            write_encoded_batches_to_wds(batch_buf, ostream)
                            batch_buf.clear()
                    if len(batch_buf) > 0:
                        write_encoded_batches_to_wds(batch_buf, ostream)

        self.encoded_tar_ready_file.touch()

    def run_all(self) -> float:
        self.make_audio_tar()
        self.make_encoded_tar()

        # k-fold:
        acc = []
        for k in self.folds:
            self.ckpt_name = f"fold_{k}_best_model.pt"
            self.ckpt_path = self.checkpoint_dir / self.ckpt_name
            self.model.reinit()
            self.train_mlp(
                self.wds_encoded_training_fold_k[k],
                [self.wds_encoded_paths_dict[k].as_posix()],
            )
            acc.append(
                self.evaluate_mlp([self.wds_encoded_paths_dict[k].as_posix()], metric=self.metric, load_ckpt=True)
            )

        for k in range(len(self.folds)):
            logger.info(f"Fold {k+1} accuracy: {acc[k]}")

        avg_acc = np.mean(acc)
        logger.info(f"The averaged accuracy of 5 folds is: {avg_acc}")

        return avg_acc
