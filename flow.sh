echo "🤖 you chose to flow with us"
echo "🤖 welcome to the joy ride"

echo "👾 running download dataset scripts"
python download_dataset.py

echo "🤖 running preprocess scripts"
python preprocess.py --decode_video --clear_corrupted --build_list

echo "👾 running train scripts"
python train.py 

echo "👾 running evaluation scripts"
python evaluate.py 

echo "🤖 yaayyy we are done!!!"
