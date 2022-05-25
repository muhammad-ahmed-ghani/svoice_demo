tr=egs/dataset/tr
ts=egs/dataset/ts
vl=egs/dataset/vl
mkdir -p $tr
mkdir -p $ts
mkdir -p $vl

CPATH=`pwd`
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/dev/mix_clean" > "egs/dataset/vl/mix.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/dev/s1" > "egs/dataset/vl/s1.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/dev/s2" > "egs/dataset/vl/s2.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/dev/s3" > "egs/dataset/vl/s3.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/dev/s4" > "egs/dataset/vl/s4.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/dev/s5" > "egs/dataset/vl/s5.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/dev/s6" > "egs/dataset/vl/s6.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/dev/s7" > "egs/dataset/vl/s7.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/dev/s8" > "egs/dataset/vl/s8.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/dev/s9" > "egs/dataset/vl/s9.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/dev/s10" > "egs/dataset/vl/s10.json"

python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/test/mix_clean" > "egs/dataset/ts/mix.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/test/s1" > "egs/dataset/ts/s1.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/test/s2" > "egs/dataset/ts/s2.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/test/s3" > "egs/dataset/ts/s3.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/test/s4" > "egs/dataset/ts/s4.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/test/s5" > "egs/dataset/ts/s5.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/test/s6" > "egs/dataset/ts/s6.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/test/s7" > "egs/dataset/ts/s7.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/test/s8" > "egs/dataset/ts/s8.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/test/s9" > "egs/dataset/ts/s9.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/test/s10" > "egs/dataset/ts/s10.json"

python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/train-360/mix_clean" > "egs/dataset/tr/mix.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/train-360/s1" > "egs/dataset/tr/s1.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/train-360/s2" > "egs/dataset/tr/s2.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/train-360/s3" > "egs/dataset/tr/s3.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/train-360/s4" > "egs/dataset/tr/s4.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/train-360/s5" > "egs/dataset/tr/s5.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/train-360/s6" > "egs/dataset/tr/s6.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/train-360/s7" > "egs/dataset/tr/s7.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/train-360/s8" > "egs/dataset/tr/s8.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/train-360/s9" > "egs/dataset/tr/s9.json"
python -m svoice.data.audio $CPATH"/Libri10Mix_Dataset/wav16k/min/train-360/s10" > "egs/dataset/tr/s10.json"
