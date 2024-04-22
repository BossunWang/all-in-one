import allin1
import os
from pathlib import Path
from pydub import AudioSegment


CWD = Path(__file__).resolve().parent


def slice():
    source_dir = "/media/bossun/新增磁碟區/Datasets/Kpop_demo"
    target_dir = "/media/bossun/新增磁碟區/Datasets/Kpop_demo_chorus_slices"

    os.makedirs(target_dir, exist_ok=True)

    for root, dirs, files in os.walk(source_dir, topdown=True):
        for name in files:
            music_path = os.path.join(root, name)

            result = allin1.analyze(music_path,
                                  include_activations=True,
                                  include_embeddings=True,
                                  # model="harmonix-fold0",
                                 )

            chorus_dict = {}
            chorus_start_sec = -1
            part_count = -1
            for segment in result.segments:
                # print(f'start: {segment.start}')
                # print(f'end: {segment.end}')
                # print(f'label: {segment.label}')

                # condition of continuos slices
                if segment.label == "chorus":
                    if chorus_start_sec == -1:
                        chorus_start_sec = segment.start
                        part_count += 1
                    chorus_end_sec = segment.end
                    chorus_dict[part_count] = [chorus_start_sec, chorus_end_sec]
                else:
                    chorus_start_sec = -1

            # select maximum interval of chorus
            max_sec = 0
            max_interval_key = ""
            for k in chorus_dict.keys():
                interval_sec = chorus_dict[k][1] - chorus_dict[k][0]
                if interval_sec > max_sec:
                    max_sec = interval_sec
                    max_interval_key = k

            print(chorus_dict[max_interval_key])
            # slice part of audio
            audio_slice = AudioSegment.from_wav(music_path)
            start_t = chorus_dict[max_interval_key][0] * 1000
            end_t = chorus_dict[max_interval_key][1] * 1000
            audio_slice = audio_slice[start_t: end_t]
            audio_slice.export(os.path.join(target_dir, name), format="wav")


if __name__ == '__main__':
    slice()