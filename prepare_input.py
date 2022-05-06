import numpy as np

#from keras.datasets import mnist
from skimage.transform import resize

import matplotlib.pyplot as plt

def visual_egomotion_input(n_patterns, visual_dim, p):
    assert visual_dim % 2 == 0

    dims = int(visual_dim/2)


    egomotion_template = np.random.choice([-1,0,1],n_patterns)
    egomotion_template = np.random.choice([-1,1],n_patterns)

    visual_input = (np.ones((n_patterns, visual_dim)).T*egomotion_template).T + np.random.choice([-1,0,1], (n_patterns, visual_dim), p=[p/2,1-p,p/2])

    #ff die max dingen

    egomotion_input = np.maximum(0, np.concatenate(([egomotion_template],[-egomotion_template]),axis=0).T)

    visual_input[:,:dims] = np.maximum(0, visual_input[:,:dims])
    visual_input[:,dims:] = np.maximum(0, -visual_input[:,dims:])


    return egomotion_input, visual_input

def visual_egomotion_input_test(visual_dim):
    assert visual_dim % 2 == 0

    dims = int(visual_dim/2)

    egomotion_input = np.repeat([[1,0]],10,axis=0)

    visual_input = np.ones((10,visual_dim))
    visual_input[:,dims:]*=0
    visual_input[2,1]=0
    visual_input[4,1]=0
    visual_input[8,1]=2

    return egomotion_input, visual_input


def visual_egomotion_input_func(patterns, t, presentation_time, fade_time):

    current_seq_id = np.floor_divide(t, presentation_time) % len(patterns)
    old_seq_id = (current_seq_id - 1 ) % len(patterns)

    fade = np.minimum(1.0,(t % presentation_time)/fade_time)

    return (patterns[current_seq_id].T * fade + patterns[old_seq_id].T * (1 - fade)).T


#Legacy
#def mouse_input(sequence, t, presentation_time=100, fade_time=30):
#    #0 = left
#    #1 = straight
#    #2 = right
#
#    images = np.array(sequence)/2
#
#    current_seq_id = np.floor_divide(t, presentation_time) % len(images)
#    old_seq_id = (current_seq_id - 1 ) % len(images)
#
#    fade = np.minimum(1.0,(t % presentation_time)/fade_time)
#
#    return images[current_seq_id] * fade + images[old_seq_id] * (1 - fade)

def mnist_images(n, size):
    (images, labels), _ = mnist.load_data()

    #Get labels of images 0-2:

    return_label_ids = np.where(labels <= 2)[0][:n]

    return_images = [ resize(images[i]/255, (size, size), anti_aliasing=True).flatten() for i in return_label_ids ]

    return return_images, return_label_ids


def mnist_input(images, t, presentation_time=100, fade_time=30):#Time is given in timesteps
    current_image_id = np.floor_divide(t, presentation_time) % len(images)
    old_image_id = (current_image_id - 1 ) % len(images)

    fade = np.min([1.0,(t % presentation_time)/fade_time])

    return images[current_image_id] * fade + images[old_image_id] * (1 - fade)





