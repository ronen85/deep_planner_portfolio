import matplotlib.pyplot as plt
import numpy as np
import torch

def show_problem_image(image, winner_idx):
    """Show image"""
    # image = image / 2 + 0.5
    npimg = image.squeeze(dim=0).numpy()
    plt.imshow(npimg)

def show_problem_images(problem_dataset, images_to_show = range(5)):
    fig = plt.figure()
    for place, idx in enumerate(images_to_show):
        sample = problem_dataset[idx]

        # print(idx, sample['image'].shape, sample['winner_idx'].shape)

        ax = plt.subplot(1, 4, place + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(idx))
        ax.axis('off')
        show_problem_image(**sample)
    plt.show()
    return

