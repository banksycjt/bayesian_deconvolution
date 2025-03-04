from numpy import array, size, flip, zeros

# add Reflective Boundary Conditions
def add_padding_reflective_BC(input_img, inference):

    padding_size = inference.padding_size
    input_img = array(input_img)
    size_x = size(input_img)[0]
    size_y = size(input_img)[1]
    
    img = zeros(3*size_x, 3*size_y)
    img[size_x:-size_x, size_y:-size_y] = input_img 
    img[:size_x, size_y:-size_y]= flip(input_img,0)
    img[-size_x:, size_y:-size_y]= flip(input_img,0)
    img[size_x:-size_x, :size_y]= flip(input_img,1)
    img[size_x:-size_x, -size_y:]= flip(input_img,1)
    img[:size_x, :size_y]= flip(flip(input_img,0),1)
    img[:size_x, -size_y:]= flip(flip(input_img,0),1)
    img[-size_x:, :size_y]= flip(flip(input_img,0),1)
    img[-size_x:, -size_y:]= flip(flip(input_img,0),1)

    return img[size_x-padding_size:-size_x+padding_size, size_y-padding_size:-size_y+padding_size]

def apply_reflective_BC_object(object, intermediate_img, padding_size):

    size_x = size(object)[0] - 2*padding_size
    size_y = size(object)[1] - 2*padding_size

    object_padding = object[padding_size:-padding_size, padding_size:-padding_size]
    
    intermediate_img[size_x:-size_x, size_y:-size_y] = object_padding 
    intermediate_img[:size_x, size_y:-size_y] = flip(object_padding, 0) 
    intermediate_img[-size_x:, size_y:-size_y] = flip(object_padding, 0) 
    intermediate_img[size_x:-size_x, :size_y] = flip(object_padding, 1) 
    intermediate_img[size_x:-size_x, -size_y:] = flip(object_padding, 1) 
    intermediate_img[:size_x, :size_y] = flip(flip(object_padding, 0), 1)
    intermediate_img[:size_x, -size_y:] = flip(flip(object_padding, 0), 1)
    intermediate_img[-size_x:, :size_y] = flip(flip(object_padding, 0), 1)
    intermediate_img[-size_x:, -size_y:] = flip(flip(object_padding, 0), 1)
    
    object = intermediate_img[size_x-padding_size:-size_x+padding_size, size_y-padding_size:-size_y+padding_size]
    
    return object

def apply_reflective_BC_shot(shot_noise_image, intermediate_img, padding_size):
    
    size_x = size(shot_noise_image)[1] - 2*padding_size
    size_y = size(shot_noise_image)[2] - 2*padding_size
    shot_noise_padding = shot_noise_image[padding_size:-padding_size, padding_size:-padding_size]
    
    intermediate_img[size_x:-size_x, size_y:-size_y] = shot_noise_padding 
    intermediate_img[:size_x, size_y:-size_y] = flip(shot_noise_padding, 0)
    intermediate_img[-size_x:, size_y:-size_y] = flip(shot_noise_padding, 0)
    intermediate_img[size_x:-size_x, :size_y] = flip(shot_noise_padding, 1)
    intermediate_img[size_x:-size_x, -size_y:] = flip(shot_noise_padding, 1)
    intermediate_img[:size_x, :size_y] = flip(flip(shot_noise_padding, 0), 1)
    intermediate_img[:size_x, -size_y:] = flip(flip(shot_noise_padding, 0), 1)
    intermediate_img[-size_x:, :size_y] = flip(flip(shot_noise_padding, 0), 1)
    intermediate_img[-size_x:, -size_y:] = flip(flip(shot_noise_padding, 0), 1) 
    
    shot_noise_image = intermediate_img[size_x-padding_size:-size_x+padding_size, size_y-padding_size:-size_y+padding_size]
    
    return shot_noise_image
