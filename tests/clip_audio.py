from pydub import AudioSegment

t1 = 30 * 1000 #Works in milliseconds
t2 = 60 * 1000
newAudio = AudioSegment.from_wav("/media/bossun/新增磁碟區/Datasets/Kpop_demo/001_TWICE_ONE_SPARK.wav")
newAudio = newAudio[t1: t2]
newAudio.export('newSong.wav', format="wav")