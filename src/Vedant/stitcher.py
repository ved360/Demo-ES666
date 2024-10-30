# class PanaromaStitcher():
#     def __init__(self):
#         pass

#     def make_panaroma_for_images_in(self,path):
#         imf = path
#         all_images = sorted(glob.glob(imf+os.sep+'*'))
#         print('Found {} Images for stitching'.format(len(all_images)))

#         ####  Your Implementation here
#         #### you can use functions, class_methods, whatever!! Examples are illustrated below. Remove them and implement yours.
#         #### Just make sure to return final stitched image and all Homography matrices from here
#         self.say_hi()
#         self.do_something()
#         self.do_something_more()

#         some_function.some_func()
#         folder_func.foo()

#         # Collect all homographies calculated for pair of images and return
#         homography_matrix_list =[]
#         # Return Final panaroma
#         stitched_image = cv2.imread(all_images[0])
#         #####
        
#         return stitched_image, homography_matrix_list 

#     def say_hi(self):
#         print('Hii From John Doe..')
    
#     def do_something(self):
#         return None
    
#     def do_something_more(self):
#         return None
    
import cv2
import numpy as np
import glob
import os

class PanaromaStitcher:
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self, path):
        # Load images
        all_images = [cv2.imread(img_path) for img_path in sorted(glob.glob(path + os.sep + '*'))]
        print('Found {} Images for stitching'.format(len(all_images)))

        # Initialize ORB detector
        orb = cv2.ORB_create()
        homography_matrix_list = []

        # Start with the first image
        stitched_image = all_images[0]

        # Iterate over pairs of images
        for i in range(len(all_images) - 1):
            img1 = all_images[i]
            img2 = all_images[i + 1]

            # Detect and compute features
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)

            # Match features using BFMatcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            # Extract matched points
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate homography (manual RANSAC if required)
            H, mask = self.estimate_homography(src_pts, dst_pts)
            homography_matrix_list.append(H)

            # Warp the next image and blend it with the previous one
            height, width = img1.shape[:2]
            result_width = width + img2.shape[1]
            stitched_image = cv2.warpPerspective(stitched_image, H, (result_width, height))
            stitched_image[0:img2.shape[0], 0:img2.shape[1]] = img2

        return stitched_image, homography_matrix_list

    def estimate_homography(self, src_pts, dst_pts):
        # Implement custom RANSAC for homography estimation if needed
        # Placeholder using OpenCV's findHomography for simplicity here
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H, mask
