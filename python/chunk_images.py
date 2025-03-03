def chunk_images(sub_raw_image_size_x, sub_raw_image_size_y, n_procs_per_dim_x, n_procs_per_dim_y,
                 padding_size, raw_image_with_padding, gain_map_with_padding, 
                 offset_map_with_padding, error_map_with_padding, processes):
    

    # Cartesian coordinates for each processor
    i_procs = (processes-1) % n_procs_per_dim_x
    j_procs = (processes-1 - i_procs) / n_procs_per_dim_y

    # Boundary coordinates of chunks or sub raw images in x an y direction
    im_raw = i_procs*sub_raw_image_size_x + 1
    ip_raw = 2*padding_size + (i_procs+1) * sub_raw_image_size_x
    jm_raw = j_procs*sub_raw_image_size_y + 1
    jp_raw = 2*padding_size + (j_procs+1) * sub_raw_image_size_y

    sub_img_size_x = ip_raw - im_raw + 1 - 2*padding_size
    sub_img_size_y = jp_raw - jm_raw + 1 - 2*padding_size

    image = raw_image_with_padding
    sub_raw_image = get_sub_image(image, im_raw, ip_raw, jm_raw, jp_raw)

    image = gain_map_with_padding
    sub_gain_map = get_sub_image(image, im_raw, ip_raw, jm_raw, jp_raw)

    image = offset_map_with_padding
    sub_offset_map = get_sub_image(image, im_raw, ip_raw, jm_raw, jp_raw)

    image = error_map_with_padding
    sub_error_map = get_sub_image(image, im_raw, ip_raw, jm_raw, jp_raw)

    image = 0

    return (sub_raw_image, sub_gain_map, sub_offset_map, sub_error_map, sub_img_size_x, sub_img_size_y, 
            im_raw, ip_raw, jm_raw, jp_raw, i_procs, j_procs)

def get_sub_image(img, im, ip, jm, jp):
	sub_img = img[im:ip, jm:jp]
	return sub_img