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

data shape
X = (27698, 128, 128, 3)
y = (27698, 40)

X_train = (22158, 128, 128, 3)
y_train = (22158, 40)

X_test = (5540, 128, 128, 3)
y_test = (5540, 40)

--------------------------------------------------

Epoch 1/10    loss: 2.1167 - accuracy: 0.4215 - val_loss: 1.2968 - val_accuracy: 0.6370
Epoch 2/10    loss: 1.1180 - accuracy: 0.6769 - val_loss: 1.4677 - val_accuracy: 0.6563
Epoch 3/10    loss: 0.7255 - accuracy: 0.7843 - val_loss: 0.6212 - val_accuracy: 0.8249
Epoch 4/10    loss: 0.5500 - accuracy: 0.8325 - val_loss: 0.7047 - val_accuracy: 0.8159
Epoch 5/10    loss: 0.4354 - accuracy: 0.8688 - val_loss: 0.6255 - val_accuracy: 0.8186
Epoch 6/10    loss: 0.3673 - accuracy: 0.8877 - val_loss: 0.2463 - val_accuracy: 0.9245
Epoch 7/10    loss: 0.3232 - accuracy: 0.9008 - val_loss: 0.3872 - val_accuracy: 0.8930
Epoch 8/10    loss: 0.2666 - accuracy: 0.9177 - val_loss: 0.2410 - val_accuracy: 0.9332
Epoch 9/10    loss: 0.2440 - accuracy: 0.9242 - val_loss: 0.3723 - val_accuracy: 0.9031
Epoch 10/10   loss: 0.2351 - accuracy: 0.9276 - val_loss: 0.2738 - val_accuracy: 0.9159


NB: Values may differ due to the shuffling of data