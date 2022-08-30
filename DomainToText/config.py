from yacs.config import CfgNode as CN

C = CN()

C.TRAIN = CN()
C.TRAIN.MAX_EPOCH = 250
C.TRAIN.BATCH_SIZE = 16
C.TRAIN.EVAL_EVERY_EPOCH = 20
C.TRAIN.WEIGHT_DECAY = 1e-6
C.TRAIN.INIT_LR = 0.0001
C.TRAIN.LR_DECAY_GAMMA = 0.1
C.TRAIN.LR_DECAY_EVAL_COUNT = 10
C.TRAIN.EARLY_STOP_EVAL_COUNT = 40

C.TRAIN.ADAM = CN()
C.TRAIN.ADAM.ALPHA = 0.8
C.TRAIN.ADAM.BETA = 0.999
C.TRAIN.ADAM.EPSILON = 1e-8
