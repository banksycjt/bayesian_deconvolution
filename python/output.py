import cv2
import matplotlib as plt

def save_data(object, object_mean, averaging_counter, inference, raw_image_size_x,raw_image_size_y, working_directory):
    
    padding_size = inference.padding_size
    gray_image = cv2.cvtColor(object[padding_size : padding_size + raw_image_size_x, padding_size : padding_size + raw_image_size_y], 
                              cv2.COLOR_BGR2GRAY)
    
    cv2.imwrite(working_directory + "inferred_object_" + str(averaging_counter) + ".tif", gray_image)
	
    gray_image = cv2.cvtColor(object_mean[padding_size : padding_size + raw_image_size_x, padding_size : padding_size + raw_image_size_y], 
                              cv2.COLOR_BGR2GRAY)

    cv2.imwrite(working_directory + "mean_inferred_object_" + str(averaging_counter) + ".tif", gray_image)
	
    return None

def plot_data(current_draw, object, mean_object, log_posterior, inference, raw_image_size_x, raw_image_size_y, raw_image_with_padding):

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    padding_size = inference.padding_size

    # 绘制第一张图
    axs[0, 0].imshow(object[padding_size: padding_size + raw_image_size_x, padding_size: padding_size + raw_image_size_y],
                     cmap='gray', origin='lower')
    axs[0, 0].set_title("Current Sample", fontsize=12)
    axs[0, 0].axis('off')  # 关闭坐标轴
    
    # 绘制第二张图
    axs[0, 1].imshow(mean_object[padding_size: padding_size + raw_image_size_x, padding_size: padding_size + raw_image_size_y],
                     cmap='gray', origin='lower')
    axs[0, 1].set_title("Mean", fontsize=12)
    axs[0, 1].axis('off')

    # 绘制第三张图

    axs[1, 0].plot(log_posterior[:current_draw], color='gray')
    axs[1, 0].set_title("log(Posterior)", fontsize=12)
    axs[1, 0].grid(True)
    axs[1, 0].legend(False)

    # 绘制第四张图
    axs[1, 1].imshow(raw_image_with_padding[padding_size: padding_size + raw_image_size_x, padding_size: padding_size + raw_image_size_y],
                     cmap='gray', origin='lower')
    axs[1, 1].set_title("Raw Image", fontsize=12)
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()
	
    return None
