import src.allin1.analyze as analyze
import src.allin1.visualize as visualize
from src.allin1.models import load_pretrained_model
import numpy as np
import time
import torch
import os
from tqdm import tqdm

FEATURE_FPS = 100


def extract(source_dir, target_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrained_model = load_pretrained_model(
        model_name='harmonix-all',
        device=device,
    )

    target_feats_dir = os.path.join(target_dir, "feats")
    target_beats_dir = os.path.join(target_dir, "beats")
    target_beats_pos_dir = os.path.join(target_dir, "beats_pos")

    os.makedirs(target_feats_dir, exist_ok=True)
    os.makedirs(target_beats_dir, exist_ok=True)
    os.makedirs(target_beats_pos_dir, exist_ok=True)

    for root, dirs, files in tqdm(os.walk(source_dir)):
        for f in files:
            audio_path = os.path.join(root, f)

            if not os.path.isfile(audio_path):
                continue

            result = analyze(audio_path,
                             include_activations=True,
                             include_embeddings=True,
                             pretrained_model=pretrained_model,
                             )

            allin1_feats = result.embeddings.mean(axis=-1).mean(axis=0)

            # mapping beat frame
            unit_feature_time = 1 / FEATURE_FPS
            feature_time_stamps = np.arange(allin1_feats.shape[0]) * unit_feature_time
            beats = np.array(result.beats)
            beat_frames = np.zeros_like(beats)

            for i, beat_time_stamp in enumerate(beats):
                closed_index = np.argmin(abs(feature_time_stamps - beat_time_stamp))
                beat_frames[i] = closed_index

            target_feats_path = os.path.join(target_feats_dir, f.replace(".wav", ".npy"))
            target_beats_path = os.path.join(target_beats_dir, f.replace(".wav", ".npy"))
            target_beats_pos_path = os.path.join(target_beats_pos_dir, f.replace(".wav", ".npy"))

            np.save(target_feats_path, allin1_feats)
            np.save(target_beats_path, beat_frames)
            np.save(target_beats_pos_path, result.beat_positions)



def main():
    source_dir = "/media/bossun/新增磁碟區/Datasets/DanceDatasets/DanceDatasets_audio"
    target_dir = "/media/bossun/新增磁碟區/Datasets/DanceDatasets/DanceDatasets_audio_allin1_data"
    extract(source_dir, target_dir)


if __name__ == '__main__':
    main()