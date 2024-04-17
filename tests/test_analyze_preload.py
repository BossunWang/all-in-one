import src.allin1.analyze as analyze
import src.allin1.visualize as visualize
from src.allin1.models import load_pretrained_model
import numpy as np
import time
import torch
from sklearn.decomposition import KernelPCA, PCA
import matplotlib.pyplot as plt
from spherecluster import VonMisesFisherMixture, SphericalKMeans
import seaborn as sns

FEATURE_FPS = 100

def calcSimilarity(beat_feats_pca_r, start_idx, intermediate_idx, end_idx, show=False):
    beat_feats_pca_unit_vec = beat_feats_pca_r / np.linalg.norm(beat_feats_pca_r, axis=-1)[:, None]

    g1_vec = np.sum(beat_feats_pca_unit_vec[0: intermediate_idx - start_idx], axis=0)
    g1_vec /= np.linalg.norm(g1_vec, axis=-1)

    g2_vec = np.sum(beat_feats_pca_unit_vec[end_idx - intermediate_idx: end_idx - start_idx], axis=0)
    g2_vec /= np.linalg.norm(g2_vec, axis=-1)
    cos_score = np.sum(g1_vec * g2_vec)

    if show:
        plt.figure(f"PCA feats")
        plt.scatter(beat_feats_pca_unit_vec[0: intermediate_idx - start_idx, 0],
                    beat_feats_pca_unit_vec[0: intermediate_idx - start_idx, 1],
                    label="g1",
                    alpha=0.3, )
        plt.scatter(beat_feats_pca_unit_vec[intermediate_idx - start_idx: end_idx - start_idx, 0],
                    beat_feats_pca_unit_vec[intermediate_idx - start_idx: end_idx - start_idx, 1],
                    label="g2",
                    alpha=0.3, )
        plt.legend()
        plt.show()
        plt.close()

    return cos_score


def runPCA(beat_feats):
    music_pca = PCA(n_components=beat_feats.shape[-1])
    beat_feats_pca = music_pca.fit_transform(beat_feats)
    expl_var_pca = np.var(beat_feats_pca, axis=0)
    expl_var_ratio_pca = expl_var_pca / np.sum(expl_var_pca)
    expl_var_ratio_pca_cumsum = expl_var_ratio_pca.cumsum()
    expl_var_ratio_pca_cond = np.where(expl_var_ratio_pca_cumsum > 0.95)
    num_components = expl_var_ratio_pca_cond[0][0]
    print(f'num_components: {num_components}')

    beat_feats_pca_r = beat_feats_pca[:, :num_components]
    return beat_feats_pca_r


def analysis_feats(all_feats, beats, beat_positions):
    feats = all_feats.mean(axis=-1).mean(axis=0)

    # mapping beat frame
    unit_feature_time = 1 / FEATURE_FPS
    total_sec = feats.shape[0] / FEATURE_FPS
    feature_time_stamps = np.arange(feats.shape[0]) * unit_feature_time
    beat_frames = np.zeros_like(beats)

    for i, beat_time_stamp in enumerate(beats):
        closed_index = np.argmin(abs(feature_time_stamps - beat_time_stamp))
        beat_frames[i] = closed_index

    beat_feat = feats[beat_frames.astype(np.int32), :]
    print(f'beat_frames: {beat_frames}')
    beat_frame_interval = beat_frames[1:] - beat_frames[:-1]
    print(f'beat_frame_interval: {beat_frame_interval}')

    total_start_time = time.time()
    beat_unit = np.max(beat_positions)
    print(f'beat_unit: {beat_unit}')

    beats_interval = (beat_frames[1:] - beat_frames[:-1]).mean() * beat_unit
    print(f'beats_interval: {beats_interval}')
    # beat_factor = int(np.ceil(5 * FEATURE_FPS / beats_interval))
    # beat_factor = min(beat_factor, 4)
    beat_factor = 1
    print(f'beat_factor: {beat_factor}')

    beat_group_frames = beat_frames[::beat_unit * beat_factor]
    print(beat_group_frames)

    cos_score_beat_unit_list = []
    cos_score_two_group_list = []
    plt.figure('cosine score highlight', figsize=(12, 2))
    for idx in range(len(beat_group_frames) - 2):
        start_idx = int(beat_group_frames[idx])
        end_idx = int(beat_group_frames[idx + 1])
        end_two_group_idx = int(beat_group_frames[idx + 2])
        beat_feats = feats[start_idx: end_two_group_idx]

        pca_start_time = time.time()
        beat_feats_pca_r = runPCA(beat_feats)
        print(f'pca elapsed time: {time.time() - pca_start_time}')

        cos_score_beat_unit_group \
            = calcSimilarity(beat_feats_pca_r, start_idx, start_idx + (end_idx - start_idx) // 2, end_idx)
        cos_score_two_group \
            = calcSimilarity(beat_feats_pca_r, start_idx, end_idx, end_two_group_idx, show=False)

        cos_score_beat_unit_list.append(cos_score_beat_unit_group)
        cos_score_two_group_list.append(cos_score_two_group)
        print(f'cos score beat unit: {cos_score_beat_unit_group}')
        print(f'cos score 2 group: {cos_score_two_group}')

        start_sec = float(start_idx) / FEATURE_FPS
        end_sec = float(end_two_group_idx) / FEATURE_FPS

        if cos_score_two_group > 0.0:
            print(f'similar segment: {start_sec}, {end_sec}')
            plt.axvspan(start_sec, end_sec, facecolor='g', alpha=0.5)
        else:
            plt.axvspan(start_sec, end_sec, facecolor='w', alpha=0.5)

        # too slow
        # cluster_start_time = time.time()
        # # movMF-soft
        # vmf_soft_g1 = VonMisesFisherMixture(n_clusters=1, posterior_type='soft')
        # vmf_soft_g1.fit(beat_feats_pca_r[0: end_idx - start_idx])
        #
        # vmf_soft_g2 = VonMisesFisherMixture(n_clusters=1, posterior_type='soft')
        # vmf_soft_g2.fit(beat_feats_pca_r[end_idx - start_idx: end_two_group_idx - start_idx])
        # print(f'cluster elapsed time: {time.time() - cluster_start_time}')
        #
        # cos_score = np.sum(vmf_soft_g1.cluster_centers_ * vmf_soft_g2.cluster_centers_)
        # print(f'cos score: {cos_score}')
        # # print(f'g1 center: {vmf_soft_g1.cluster_centers_}')
        # # print(f'g2 center: {vmf_soft_g2.cluster_centers_}')
        # # print(f'g1 concentrations_: {vmf_soft_g1.concentrations_}')
        # # print(f'g2 concentrations_: {vmf_soft_g2.concentrations_}')

    print(f'total elapsed time: {time.time() - total_start_time}')
    plt.savefig("cos_timeline")
    plt.close()

    plt.figure('cosine score beat unit distribution')
    sns.displot(np.array(cos_score_beat_unit_list))
    plt.savefig("cos_beat_unit_dist")
    plt.close()

    plt.figure('cosine score two group distribution')
    sns.displot(np.array(cos_score_two_group_list))
    plt.savefig("cos_two_group_dist")
    plt.close()


def test_analyze():
    # music_path = "/media/bossun/新增磁碟區/Datasets/DanceDatasets/DanceDatasets_audio/delight/delight_081_20231017_1#0-213#0.wav"  # 212sec
    # music_path = "/media/bossun/新增磁碟區/Datasets/CHT2024NewYearMediaDinner/CHT2024NewYearMediaDinner_001.wav" # 15sec
    # music_path = "/media/bossun/新增磁碟區/Datasets/DeepMotionDataset_wavs/Swhite_001_EightThreOne_EastSideEastSide_10s_40s_customModel_9srfi49avS36uUqXT8rhxb.wav"  # 30sec
    # music_path = "/media/bossun/新增磁碟區/Datasets/中華一番/君さえいれば.wav"
    # music_path = "/media/bossun/新增磁碟區/Datasets/KpopTop_test/010_NMIXX_Breaker_.wav"
    # music_path = "/media/bossun/新增磁碟區/Datasets/Jpop_audio/世界は恋に落ちている踊ってみた.wav"
    music_path = "/media/bossun/新增磁碟區/Datasets/Kpop_demo/001_TWICE_ONE_SPARK_0s_60s.wav"
    # music_path = "/media/bossun/新增磁碟區/Datasets/Kpop_demo/002_LE_SSERAFIM_Perfect_Night_0s_60s.wav"

    start_time = time.time()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrained_model = load_pretrained_model(
        model_name='harmonix-all',
        device=device,
    )

    result = analyze(music_path,
                     include_activations=True,
                     include_embeddings=True,
                     pretrained_model=pretrained_model,
                     )
    # visualize(
    #     result,
    #     out_dir='./viz',
    # )

    end_time = time.time()
    print(f'elapsed time: {end_time - start_time}')
    print(f'bpm: {result.bpm}')
    print(f'beat: {np.array(result.beats).shape}')
    print(f'down beat: {np.array(result.downbeats).shape}')
    print(f'beat_positions: {np.array(result.beat_positions)}')
    print(f'act beat: {np.array(result.activations["beat"]).shape}')
    print(f'embeddings: {result.embeddings.shape}')
    print(f'segments: {result.segments}')

    # for segment in result.segments:
    #     print(f'start: {segment.start}')
    #     print(f'end: {segment.end}')
    #     print(f'label: {segment.label}')

    analysis_feats(np.array(result.embeddings),
                   np.array(result.beats),
                   np.array(result.beat_positions))


def main():
    test_analyze()


if __name__ == '__main__':
    main()