import math
import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import config
from model import Discriminator, Generator
from utils import (
    gradient_penalty,
    plot_to_tensorboard,
    save_checkpoint,
    load_checkpoint,
    generate_examples,
)

torch.backends.cudnn.benchmarks = True


def get_loaders(img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(
            [0.5 for _ in range(config.IMG_CHANNELS)],
            [0.5 for _ in range(config.IMG_CHANNELS)],
        )
    ])
    batch_size = config.BATCH_SIZES[int(math.log2(img_size / 4))]
    dataset = datasets.ImageFolder(root=config.DATASET, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )
    return loader, dataset


def train_fn(
    critic,
    gen,
    dataloader,
    dataset,
    step,
    alpha,
    opt_critic,
    opt_gen,
    tensorboard_step,
    writer,
    scaler_gen,
    scaler_critic,
):
    tk = tqdm(dataloader)
    for batch_idx, (real, _) in enumerate(tk):
        real = real.to(config.DEVICE)
        curr_batch_size = real.shape[0]

        # Train critic ==> max { E[critic(real)] - E[critic(fake)] }  <-> min -{ E[critic(real)] - E[critic(fake)] }
        noise = torch.randn(curr_batch_size, config.Z_DIM, 1, 1).to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake = gen(noise, alpha, step)
            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)
            gp = gradient_penalty(critic, real, fake, alpha, step, device=config.DEVICE)
            loss_critic = (
                - (torch.mean(critic_real) - torch.mean(critic_fake))
                + config.LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )

        opt_critic.zero_grad()
        scaler_critic.scale(loss_critic).backward()
        scaler_critic.step(opt_critic)
        scaler_critic.update()

        # Train Generator ==> max { E[critic(gen_fake)] } <-> min -{ E[critic(gen_fake)] }
        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, alpha, step)
            loss_gen = - torch.mean(gen_fake)

        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        # Update alpha and ensure less than 1
        alpha += curr_batch_size / ((config.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset))
        alpha = min(alpha, 1)

        if batch_idx % 500 == 0:
            with torch.no_grad():
                fixed_fakes = gen(config.FIXED_NOISE, alpha, step) * 0.5 + 0.5
            plot_to_tensorboard(
                writer,
                loss_critic.item(),
                loss_gen.item(),
                real.detach(),
                fixed_fakes.detach(),
                tensorboard_step,
            )
            tensorboard_step += 1

        tk.set_postfix(gp=gp.item(), loss_critic=loss_critic.item())
        return tensorboard_step, alpha


def main():
    gen = Generator(
        config.Z_DIM, config.IN_CHANNELS, img_channels=config.IMG_CHANNELS).to(config.DEVICE)
    critic = Discriminator(
        config.IN_CHANNELS, img_channels=config.IMG_CHANNELS).to(config.DEVICE)

    # initialize optimizers and scalers for FP16 training
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    scaler_critic = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()

    # for tensorboard plotting
    writer = SummaryWriter(f"logs/ProgressiveGAN")

    if config.LOAD_MODEL:
        load_checkpoint(config.GEN_CHECKPOINT, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CRITIC_CHECKPOINT, critic, opt_critic, config.LEARNING_RATE)

    gen.train()
    critic.train()

    tensorboard_step = 0
    step = int(math.log2(config.INITIAL_IMG_SIZE / 4))
    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5   # start with very low alpha
        dataloader, dataset = get_loaders(4 * 2 ** step)   # 4->0, 8->1, 16->2, ...
        print(f"Current image size: {4 * 2 ** step}")

        for epoch in range(num_epochs):
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            tensorboard_step, alpha = train_fn(
                critic,
                gen,
                dataloader,
                dataset,
                step,
                alpha,
                opt_critic,
                opt_gen,
                tensorboard_step,
                writer,
                scaler_gen,
                scaler_critic,
            )

            if config.SAVE_MODEL:
                save_checkpoint(gen, opt_gen, filename=config.GEN_CHECKPOINT)
                save_checkpoint(critic, opt_critic, filename=config.CRITIC_CHECKPOINT)

        step += 1  # progress to the next img size


if __name__ == "__main__":
    main()