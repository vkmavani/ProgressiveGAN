import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET = "data/celeba"
SAVE_MODEL = True
LOAD_MODEL = False
GEN_CHECKPOINT = "generator.pth"
CRITIC_CHECKPOINT = "discriminator.pth"

INITIAL_IMG_SIZE = 4
TARGET_IMG_SIZE = 512
IMG_CHANNELS = 3
Z_DIM = 256          # In paper -> 512
IN_CHANNELS = 256    # In paper -> 512

LEARNING_RATE = 1e-3
BATCH_SIZES = [16, 16, 16, 16, 16, 16, 16, 8, 4]     # change depending on the machine's vram
LAMBDA_GP = 10

UPDATE_STEPS = 50    # this is used to update the image size after that much steps
PROGRESSIVE_EPOCHS = [UPDATE_STEPS] * len(BATCH_SIZES)
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 4
