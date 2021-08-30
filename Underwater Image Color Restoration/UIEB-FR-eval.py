import tensorflow as tf
from sklearn.metrics import mean_squared_error
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-folder', default='DCP/OutputImages/UIEB', help='Images To Be Evaluated')
    parser.add_argument('--suffix', default='_DCP.jpg', help='Suffix of Candidate Images Compared to Original Image Name')
    parser.add_argument('--score', default='MSE_DCP.log', help='Write PSNR \t SSIM to This File')
    args = parser.parse_args()

    ref_image_folder = "reference-890/"
    score = open(args.score, mode = 'a',encoding='utf-8')
    print("PSNR \t SSIM", file=score)
    for image_name in os.listdir(ref_image_folder):
        candidate_image_name = image_name.split('.')[0] + args.suffix
        candidate_image_path = os.path.join(args.image_folder, candidate_image_name)
        if os.path.exists(candidate_image_path):
            candidate_image = tf.io.read_file(candidate_image_path)
            candidate_image = tf.io.decode_image(candidate_image) # [480, 640, 3]
            candidate_image = tf.image.convert_image_dtype(candidate_image, tf.float32)
            candidate_image = tf.image.resize(candidate_image, [480, 640]) # []480, 640, 3] float32

            ref_image_path = os.path.join(ref_image_folder, image_name)
            ref_image = tf.io.read_file(ref_image_path)
            ref_image = tf.io.decode_image(ref_image) # [1200, 1600, 3]
            ref_image = tf.image.convert_image_dtype(ref_image, tf.float32)
            ref_image = tf.image.resize(ref_image, [480, 640]) # []480, 640, 3] float32
            
            # PSNR = tf.image.psnr(candidate_image, ref_image, max_val=1.0)
            # SSIM = tf.image.ssim(candidate_image, ref_image, max_val=1.0)
            # print(image_name, file=score)
            # print("{0}\t{1}".format(PSNR, SSIM), file=score)

            MSE = (mean_squared_error(candidate_image[:,:,0], ref_image[:,:,0]) + \
                mean_squared_error(candidate_image[:,:,1], ref_image[:,:,1]) + \
                mean_squared_error(candidate_image[:,:,2], ref_image[:,:,2]))/3
            print(image_name, file=score)
            print(MSE, file=score)
            
    print("Done", file=score)
    score.close()



