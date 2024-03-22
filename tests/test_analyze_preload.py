import allin1.analyze as analyze
from allin1.models import load_pretrained_model
import numpy as np
import time
import torch


def test_analyze():
    # music_path = "/media/bossun/新增磁碟區/Datasets/DanceDatasets/DanceDatasets_audio/delight/delight_081_20231017_1#0-213#0.wav"  # 212sec
    # music_path = "/media/bossun/新增磁碟區/Datasets/CHT2024NewYearMediaDinner/CHT2024NewYearMediaDinner_001.wav" # 15sec
    # music_path = "/media/bossun/新增磁碟區/Datasets/DeepMotionDataset_wavs/Swhite_001_EightThreOne_EastSideEastSide_10s_40s_customModel_9srfi49avS36uUqXT8rhxb.wav"  # 30sec
    # music_path = "/media/bossun/新增磁碟區/Datasets/中華一番/君さえいれば.wav"
    music_path = "/media/bossun/新增磁碟區/Datasets/KpopTop_test/010_NMIXX_Breaker_.wav"

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
    end_time = time.time()
    print(f'elapsed time: {end_time - start_time}')
    print(f'bpm: {result.bpm}')
    print(f'beat: {np.array(result.beats).shape}')
    print(f'act beat: {np.array(result.activations["beat"]).shape}')
    print(f'embeddings: {result.embeddings.shape}')
    print(f'segments: {result.segments}')

    # for segment in result.segments:
    #     print(f'start: {segment.start}')
    #     print(f'end: {segment.end}')
    #     print(f'label: {segment.label}')


def main():
    test_analyze()


if __name__ == '__main__':
    main()