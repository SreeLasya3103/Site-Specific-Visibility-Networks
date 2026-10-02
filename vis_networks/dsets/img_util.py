import torchvision.transforms as tf
import torchvision.transforms.functional as tff

def get_resize_crop_fn(dim):
    def resize_crop(image):
        #height over width
        target_ratio = dim[0] / dim[1]
        ratio = image.shape[-2] / image.shape[-1]
        
        #if the the image is too tall, crop the top and bottom
        #otherwise crop sides
        if ratio > target_ratio:
            crop = (round(target_ratio*image.shape[-1]), image.shape[-1])
        else:
            crop = (image.shape[-2], round(image.shape[-2]/target_ratio))

        image = tff.center_crop(image, crop)
        image = tff.resize(image, dim, tf.InterpolationMode.BICUBIC)

        return image
    
    return resize_crop