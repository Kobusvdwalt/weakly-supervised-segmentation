
import cv2

def plot_masks():
    image_name = '2008_004365'
    image = cv2.imread(f'../datasets/voc2012/JPEGImages/{image_name}.jpg') / 255.0
    mask = cv2.imread(f'../datasets/voc2012/SegmentationClass/{image_name}.png') / 255.0

    alpha = 0.6
    comb = alpha * mask + (1- alpha) * image

    cv2.imwrite(f'{image_name}_image.png', image * 255)
    cv2.imwrite(f'{image_name}_mask.png', mask * 255)
    cv2.imwrite(f'{image_name}_comb.png', comb * 255)
    cv2.waitKey()


plot_masks()
