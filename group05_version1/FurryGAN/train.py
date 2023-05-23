import torch
import matplotlib.pyplot as plt

from losses import *
from utils import set_requires_gradient, mask_down_sample


def draw_loss_plots(step_size, generator_losses, discriminator_losses):
    x_iters = [iter_idx * step_size for iter_idx in range(len(generator_losses))]
    plt.plot(x_iters, generator_losses, '-b', label='generator loss')
    plt.plot(x_iters, discriminator_losses, '-r', label='discriminator loss')

    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(loc='upper left')
    plt.title("Loss Plot")

    plt.show()


def train(
    fg_generator,
    mask_generator,
    bg_generator,
    discriminator,
    g_optim,
    d_optim,
    train_loader,
    args,
):
    # args.logger.info("### Training has started! ###")

    device = args.device
    gamma = args.gamma_start
    gamma_every = 5000 / ((args.gamma_max - args.gamma_start) / args.gamma_step)
    lambda_coarse = args.lambda_coarse
    lambda_fine = args.lambda_fine
    lambda_step = (args.lambda_coarse - args.lambda_min) / 5000
    train_generator = iter(train_loader)
    iter_idx = 0

    generator_loss_list = []
    discriminator_loss_list = []

    while True:
        ####################### Discriminator Training ########################

        # Freeze generators, unfreeze discriminator
        set_requires_gradient(fg_generator, False)
        set_requires_gradient(mask_generator, False)
        set_requires_gradient(bg_generator, False)
        set_requires_gradient(discriminator, True)

        # Check if the iteration variable reached the limit
        if iter_idx == args.num_iters:
            break

        try:
            # Get the next batch
            image_batch, mask_batch = next(train_generator)
        except StopIteration:
            # If the generator reached the end, restart it
            train_generator = iter(train_loader)
            image_batch, mask_batch = next(train_generator)

        # Move tensors to device
        image_batch = image_batch.to(args.device)
        mask_batch = mask_batch.to(args.device)

        # Get batch size
        batch_size = image_batch.size(dim=0)

        # Sample two fg latent codes, each with size (batch_size/2, latent_size)
        z_fg = torch.randn(int(batch_size / 2), args.latent_size, device=args.device)
        z_comp_fg = torch.randn(
            int(batch_size / 2), args.latent_size, device=args.device
        )

        # Sample bg latent code as front part of the z_comp_fg
        bg_latent_size = int(args.latent_size / 4)
        z_bg = z_comp_fg[:, :bg_latent_size]

        # Run foreground generator
        f_fg, s_fg, x_fg = fg_generator(z_fg)
        f_comp_fg, s_comp_fg, x_comp_fg = fg_generator(z_comp_fg)

        # Run mask generator
        comp_mask, _, _ = mask_generator(s_comp_fg, f_comp_fg, gamma)
        fg_mask, _, _ = mask_generator(s_fg, f_fg, gamma)

        # Run background generator
        _, _, x_bg = bg_generator(z_bg)

        # Generate composite images
        x_comp = comp_mask * x_comp_fg + (1 - comp_mask) * x_bg

        # Create fake and real images
        real_images = image_batch
        fake_images = torch.concat((x_fg, x_comp))
        concat_images = torch.concat((real_images, fake_images))
        image_labels = torch.hstack((torch.ones(batch_size), torch.zeros(batch_size)))

        # Run discriminator
        binary_preds, mask_preds = discriminator(concat_images)

        # Calculate discriminator loss
        mask_labels = torch.concat((fg_mask, comp_mask))
        L_adv = d_loss(image_labels, binary_preds)
        L_pred = mask_prediction_loss(mask_labels, mask_preds[:batch_size])
        discriminator_loss = L_adv + L_pred

        # Update discriminator params
        discriminator.zero_grad()
        discriminator_loss.backward()
        d_optim.step()

        # R1 regularization loss
        if iter_idx % args.reg_every == 1:
            real_images.requires_grad = True

            # Generate discriminator preds for real images
            real_preds, _ = discriminator(real_images)

            # Calculate R1 regularization loss
            L_R1 = r1_loss(real_preds, real_images)

            # Update discriminator params
            discriminator.zero_grad()
            # TODO check this part
            (args.r1 / 2 * L_R1 * args.reg_every + 0 * real_preds[0]).backward()
            d_optim.step()

        # Increment the iteration variable
        iter_idx += 1

        ####################### Generator Training #######################

        # Freeze discriminator, unfreeze generators
        set_requires_gradient(fg_generator, True)
        set_requires_gradient(mask_generator, True)
        set_requires_gradient(bg_generator, True)
        set_requires_gradient(discriminator, False)

        # Check if the iteration variable reached the limit
        if iter_idx == args.num_iters:
            break

        try:
            # Get the next batch
            image_batch, mask_batch = next(train_generator)
        except StopIteration:
            # If the generator reached the end, restart it
            train_generator = iter(train_loader)
            image_batch, mask_batch = next(train_generator)

        # Move tensors to device
        image_batch = image_batch.to(args.device)
        mask_batch = mask_batch.to(args.device)

        # Get batch size
        batch_size = image_batch.size(dim=0)

        # Sample two fg latent codes, each with size (batch_size/2, latent_size)
        z_fg = torch.randn(int(batch_size / 2), args.latent_size, device=args.device)
        z_comp_fg = torch.randn(
            int(batch_size / 2), args.latent_size, device=args.device
        )

        # Sample bg latent code as front part of the z_comp_fg
        bg_latent_size = int(args.latent_size / 4)
        z_bg = z_comp_fg[:, :bg_latent_size]

        # Run foreground generator
        f_fg, _, x_fg = fg_generator(z_fg)
        f_comp_fg, s_comp_fg, x_comp_fg = fg_generator(z_comp_fg)

        # Run mask generator
        m, m_coarse, m_fine = mask_generator(s_comp_fg, f_comp_fg, gamma)

        # Run background generator
        _, _, x_bg = bg_generator(z_bg)

        # Generate composite images
        x_comp = m * x_comp_fg + (1 - m) * x_bg

        # Create fake and real images
        real_images = image_batch
        fake_images = torch.concat((x_fg, x_comp))
        fake_labels = torch.zeros(batch_size)
        real_labels = torch.ones(batch_size)

        # Run discriminator
        binary_preds, mask_preds = discriminator(fake_images)

        # Calculate generator loss
        L_adv = g_loss(binary_preds)
        L_cons = mask_consistency_loss(mask_preds[int(batch_size / 2):], mask_preds[:int(batch_size / 2)])
        L_coarse = coarse_mask_loss(m_coarse, args.phi1, args.device)
        m_fine = m - m_coarse
        L_fine = fine_mask_loss(m_fine, args.phi2, args.device)
        L_reg = background_loss(x_comp, x_bg)
        generator_loss = L_adv + L_cons + lambda_coarse * L_coarse + lambda_fine * L_fine + L_reg

        # Update generator params (g_optim has parameters of all generators)
        fg_generator.zero_grad()
        bg_generator.zero_grad()
        mask_generator.zero_grad()
        generator_loss.backward()
        g_optim.step()

        # Increment the iteration variable
        iter_idx += 1

        # Update gamma and lambda
        if iter_idx % gamma_every == 0 and gamma < args.gamma_max:
            gamma += args.gamma_step
        if lambda_coarse > args.lambda_min:
            lambda_coarse -= lambda_step
            lambda_fine += lambda_step
        if iter_idx % 100 == 0:
            print(
                "Iteration: %d, Generator Loss: %.4f, Discriminator Loss: %.4f"
                % (iter_idx, generator_loss.item(), discriminator_loss.item())
            )
            generator_loss_list.append(generator_loss.item())
            discriminator_loss_list.append(discriminator_loss.item())

    draw_loss_plots(step_size=100, generator_losses=generator_loss_list, 
                    discriminator_losses=discriminator_loss_list)