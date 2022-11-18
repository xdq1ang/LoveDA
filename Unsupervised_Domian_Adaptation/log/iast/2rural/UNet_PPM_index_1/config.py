from configs.ToRURAL import SOURCE_DATA_CONFIG,TARGET_DATA_CONFIG, EVAL_DATA_CONFIG, TARGET_SET
MODEL = 'ResNet'


IGNORE_LABEL = -1
MOMENTUM = 0.9
NUM_CLASSES = 7

SAVE_PRED_EVERY = 2000
SNAPSHOT_DIR = './log/iast/2rural/dev'

#Hyper Paramters
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 1e-2
LEARNING_RATE_D = 1e-4
NUM_STEPS = 20000
NUM_STEPS_STOP = 20000  # Use damping instead of early stopping
PREHEAT_STEPS = int(NUM_STEPS / 20)
POWER = 0.9
EVAL_EVERY=500

DISCRIMINATOR = dict(
    lambda_kldreg_weight=0.1,
    lambda_entropy_weight=3.0,
    weight=0.05
)
GENERATE_PSEDO_EVERY = 500
WARMUP_STEP = 5000
PSEIDO_DICT = dict(
    pl_alpha=0.2,
    pl_gamma=8.0,
    pl_beta=0.9
)
PSEUDO_LOSS_WEIGHT = 0.5
SOURCE_LOSS_WEIGHT = 1.0

TARGET_SET = TARGET_SET
SOURCE_DATA_CONFIG=SOURCE_DATA_CONFIG
TARGET_DATA_CONFIG=TARGET_DATA_CONFIG
EVAL_DATA_CONFIG=EVAL_DATA_CONFIG
TTA_CFG = None