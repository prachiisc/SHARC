for audio in $(ls /home1/somil/language_diarization/Displace_data/eval_1/*/*.wav);  
do
python3 /home1/somil/VBx-VAD/VAD.py --in-audio=$audio --in-VAD=Pyannote_VAD;  
done

for audio in $(ls /home1/somil/language_diarization/Displace_data/eval_2/*/*.wav);  
do
python3 /home1/somil/VBx-VAD/VAD.py --in-audio=$audio --in-VAD=Pyannote_VAD;  
done
