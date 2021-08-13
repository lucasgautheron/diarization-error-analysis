import os

wavs = os.listdir('output/clips')

process = []
for wav in wavs:
    name, ext = os.path.splitext(wav)
    
    if ext.lower() != '.wav':
        continue

    if not os.path.exists(os.path.join('output/scores', name)):
        process.append(wav)

with open('output/clips/vtc.txt', 'w+') as fp:
    fp.write("\n".join(process))
