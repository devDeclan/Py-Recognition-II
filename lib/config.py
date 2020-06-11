import os

DATASET_ROOT = "dataset"
FRAMES_ROOT = os.path.join(DATASET_ROOT, "frames")
VIDEOS_ROOT = os.path.join(DATASET_ROOT, "videos")
ANNOTATIONS_ROOT = os.path.join(DATASET_ROOT, "annotations")

EPOCHS = 10
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
WORKERS = 8

MODEL_ROOT = "model"

FRAME_RATE = 5
VIDEOS_PER_CLASS = 20
FRAMES_PER_VIDEO = 20

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