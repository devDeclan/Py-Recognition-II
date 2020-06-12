# Py-Recognition-II

--------------------------------------------------

### Running the model trainer
assumption is that you are on a linux subsystem with apt available

create an environment with the neccessary packages by running
$ sudo bash install.sh

dataset can be download with the helper script download_dataset.py by running
$ python download_dataset.py

generate frames and neccessary files by running
$ python preprocess.py --decode_video --clear_corrupted --build_list

train the model by running
$ python train.py

you can evaluate the model by running
$ python evaluate

models are stored in models/ directory
dataset is stored in dataset/ directory

--------------------------------------------------

AWS t2.2xlarge
ubuntu 18.04 LTS
vCPUs 8
arm64
Memory 16GB
Storage 100GB

--------------------------------------------------

EPOCHS = 10
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
WORKERS = 8

MODEL_ROOT = "model"

CLASSES = [
	"Archery",
	"BalanceBeam",
	"BaseballPitch",
	"Basketball",
	"Biking",
	"Bowling",
	"BoxingPunchingBag",
	"BoxingSpeedBag",
	"BreastStroke",
	"CliffDiving",
	"CricketBowling",
	"CricketShot",
	"Diving",
	"Fencing",
	"FieldHockeyPenalty",
	"GolfSwing",
	"HammerThrow",
	"HighJump",
	"HorseRace",
	"IceDancing",
	"JavelinThrow",
	"JugglingBalls",
	"JumpRope",
	"Kayaking",
	"LongJump",
	"Lunges",
	"ParallelBars",
	"PoleVault",
	"Rafting",
	"Rowing",
	"Shotput",
	"SkateBoarding",
	"Skiing",
	"SkyDiving",
	"SoccerJuggling",
	"SoccerPenalty",
	"Surfing",
	"TableTennisShot",
	"TennisSwing",
	"VolleyballSpiking",
]

--------------------------------------------------

Total params: 2,822,624
Trainable params: 2,813,888
Non-trainable params: 8,736

--------------------------------------------------

(27698, 128, 128, 3)
(27698, 40)

(22158, 128, 128, 3)
(22158, 40)

(5540, 128, 128, 3)
(5540, 40)

--------------------------------------------------

Epoch 1/10
	loss: 2.1444
	accuracy: 0.4080
	val_loss: 1.4796
	val_accuracy: 0.5682

Epoch 2/10
	loss: 1.1553
	accuracy: 0.6656
	val_loss: 0.6885
	val_accuracy: 0.8000

Epoch 3/10
	loss: 0.7696
	accuracy: 0.7693
	val_loss: 0.7628
	val_accuracy: 0.7850

Epoch 4/10
	loss:
	accuracy:
	val_loss:
	val_accuracy:
	
Epoch 5/10
	loss:
	accuracy:
	val_loss:
	val_accuracy:
	
Epoch 6/10
	loss:
	accuracy:
	val_loss:
	val_accuracy:
	
Epoch 7/10
	loss:
	accuracy:
	val_loss:
	val_accuracy:
	
Epoch 8/10
	loss:
	accuracy:
	val_loss:
	val_accuracy:
	
Epoch 9/10
	loss:
	accuracy:
	val_loss:
	val_accuracy:
	
Epoch 10/10
	loss:
	accuracy:
	val_loss:
	val_accuracy:
	