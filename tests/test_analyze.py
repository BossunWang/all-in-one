import math
import allin1
import numpy as np
import time
from pathlib import Path
import torchaudio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import ticker
from sklearn import manifold
from sdtw_div.numba_ops import sdtw_div, sharp_sdtw_div
import sys

CWD = Path(__file__).resolve().parent


HARMONIX_COLORS = {
  'start': 0,
  'end': 0,
  'intro': 1,
  'outro': 1,
  'break': 2,
  'bridge': 2,
  'inst': 3,
  'solo': 3,
  'verse': 4,
  'chorus': 5,
}

labels = ['start, end', 'intro, outro', 'break, bridge', 'inst, solo', 'verse', 'chorus']

def add_2d_scatter(ax, points, points_color, classes, title=None):
    colors_map = ListedColormap(['r', 'b', 'g'])
    x, y = points.T
    scatter = ax.scatter(x, y, c=points_color, s=2, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.legend(handles=scatter.legend_elements()[0], labels=classes)


def plot_2d(points, points_color, classes, title, save_path=None):
    fig, ax = plt.subplots(facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color, classes)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def test_analyze():
    # music_path = "/media/bossun/新增磁碟區/Datasets/DanceDatasets/DanceDatasets_audio/delight/delight_081_20231017_1#0-213#0.wav"  # 212sec
    # music_path = "/media/bossun/新增磁碟區/Datasets/CHT2024NewYearMediaDinner/CHT2024NewYearMediaDinner_001.wav" # 15sec
    # music_path = "/media/bossun/新增磁碟區/Datasets/DeepMotionDataset_wavs/Swhite_001_EightThreOne_EastSideEastSide_10s_40s_customModel_9srfi49avS36uUqXT8rhxb.wav"  # 30sec
    # music_path = "/media/bossun/新增磁碟區/Datasets/中華一番/君さえいれば.wav"
    music_path = "/media/bossun/新增磁碟區/Datasets/KpopTop_test/010_NMIXX_Breaker_.wav"
    # set to 2 for testing elapsed time after second inference
    for t in range(1):
        start_time = time.time()
        result = allin1.analyze(music_path,
                              include_activations=True,
                              include_embeddings=True,
                              # model="harmonix-fold0",
                             )
        end_time = time.time()
        print(f'elapsed time: {end_time - start_time}')
        fig = allin1.visualize(result, multiprocess=False, out_dir='./viz',)

    print(f'bpm: {result.bpm}')
    print(f'beat: {np.array(result.beats).shape}')
    print(f'act beat: {np.array(result.activations["beat"]).shape}')
    print(f'embeddings: {result.embeddings.shape}')
    print(f'segments: {result.segments}')

    # for segment in result.segments:
    #     print(f'start: {segment.start}')
    #     print(f'end: {segment.end}')
    #     print(f'label: {segment.label}')

    allin1_feats = result.embeddings.mean(axis=-1).mean(axis=0)
    group_labels = np.zeros(len(allin1_feats))
    group_order_labels = np.zeros(len(allin1_feats))
    FPS = 100
    for frame in range(group_labels.shape[0]):
        for gi, segment in enumerate(result.segments):
            if segment.start <= frame / FPS <= segment.end:
                group_labels[frame] = HARMONIX_COLORS[segment.label]
                group_order_labels[frame] = gi
                break

    # t_sne_allin1 = manifold.TSNE(
    #     n_components=2,
    # )
    #
    # feats_allin1_tsne = t_sne_allin1.fit_transform(allin1_feats)
    # classes = [labels[int(idx)] for idx in set(group_labels)]
    # plot_2d(feats_allin1_tsne, group_labels, classes, "allin1 features", "allin1_feats_tsne.jpg")

    # analysis group features similarity
    ref_group_idx = 3
    ref_feats = allin1_feats[group_order_labels == ref_group_idx]

    # find similar group
    min_sdtw_value = sys.float_info.max
    for gi in set(group_order_labels):
        group_feats = allin1_feats[group_order_labels == gi]
        sdtw_value = sdtw_div(ref_feats, group_feats, gamma=1.0)
        sharp_sdtw_value = sharp_sdtw_div(ref_feats, group_feats, gamma=1.0)
        print(f'allin1: {ref_group_idx} vs {gi}: {sdtw_value + sharp_sdtw_value}')

        if min_sdtw_value > sdtw_value + sharp_sdtw_value and gi != ref_group_idx:
            min_sdtw_value = sdtw_value + sharp_sdtw_value
            min_gi = gi

    print(f'allin1: {ref_group_idx} similar to {min_gi}')

    # -----------------------MERT analysis-----------------------------------
    # load MERT feats
    MERT_feats_org = np.load("music_feats.npy")
    MERT_feats = MERT_feats_org.squeeze(0).reshape(-1, MERT_feats_org.shape[-1])

    # analysis tsne via group labels
    uint_size = 150
    FPS = 30
    group_labels = np.zeros((len(MERT_feats)))
    group_order_labels = np.zeros(len(MERT_feats))

    for idx, f in enumerate(MERT_feats):
        group_idx = math.floor(idx / uint_size)
        unit_group_idx = idx % uint_size
        frame = group_idx * uint_size / 2 + unit_group_idx if group_idx > 0 else unit_group_idx
        frame = int(frame)

        for gi, segment in enumerate(result.segments):
            if segment.start <= frame / FPS <= segment.end:
                group_labels[frame] = HARMONIX_COLORS[segment.label]
                group_order_labels[frame] = gi
                break

    # fade in / out
    MERT_feats_fade = MERT_feats_org.squeeze(0).copy()
    for idx in range(len(MERT_feats_fade) - 1):
        o1 = MERT_feats_fade[idx, uint_size // 2:]
        o2 = MERT_feats_fade[idx + 1, :uint_size // 2]

        fade_in_weight = np.linspace(0, 1, uint_size // 2)
        fade_out_weight = np.linspace(1, 0, uint_size // 2)
        fade_feat = o1 * fade_out_weight[:, None] + o2 * fade_in_weight[:, None]

        MERT_feats_fade[idx, uint_size // 2:] = fade_feat
        MERT_feats_fade[idx + 1, :uint_size // 2] = fade_feat

        # cos_sim = (o1 * o2).sum(axis=-1) / (np.linalg.norm(o1, axis=-1) * np.linalg.norm(o2, axis=-1))
        # print(cos_sim.mean())

    MERT_feats_fade = MERT_feats_fade.reshape(-1, MERT_feats_fade.shape[-1])

    # analysis group features similarity
    ref_group_idx = 3
    ref_feats = MERT_feats_fade[group_order_labels == ref_group_idx]
    # find similar group
    min_sdtw_value = sys.float_info.max
    for gi in set(group_order_labels):
        group_feats = MERT_feats_fade[group_order_labels == gi]
        sdtw_value = sdtw_div(ref_feats, group_feats, gamma=1.0)
        sharp_sdtw_value = sharp_sdtw_div(ref_feats, group_feats, gamma=1.0)
        print(f'MERT: {ref_group_idx} vs {gi}: {sdtw_value + sharp_sdtw_value}')

        if min_sdtw_value > sdtw_value + sharp_sdtw_value and gi != ref_group_idx:
            min_sdtw_value = sdtw_value + sharp_sdtw_value
            min_gi = gi

    print(f'MERT: {ref_group_idx} similar to {min_gi}')

    # TSNE MERT features
    t_sne_MERT = manifold.TSNE(
        n_components=2,
    )
    feats_tsne = t_sne_MERT.fit_transform(MERT_feats_fade)
    # np.save("tsne_feats.npy", feats_tsne)
    # feats_tsne = np.load("tsne_feats.npy")
    classes = [labels[int(idx)] for idx in set(group_labels)]
    plot_2d(feats_tsne, group_labels, classes, "MERT features", "MERT_feats_tsne.jpg")


def main():
    test_analyze()


if __name__ == '__main__':
    main()