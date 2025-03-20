import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

selfie_seg = mp.solutions.selfie_segmentation
segment = selfie_seg.SelfieSegmentation(model_selection=1)  # Modelo mais preciso

def modifyBackground(image, background_image=None, blur=25, threshold=0.3, display=True, method='changeBackground'):
    # Convert the input image from BGR to RGB format.
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the segmentation.
    result = segment.process(RGB_img)
    
    # Get a binary mask having pixel value 1 for the object and 0 for the background.
    binary_mask = result.segmentation_mask > threshold
    
    # Stack the same mask three times to make it a three-channel image.
    binary_mask_3 = np.dstack((binary_mask, binary_mask, binary_mask))
    

        
    if method == 'blurBackground':
        # Create a blurred copy of the input image.
        blurred_image = cv2.GaussianBlur(image, (blur, blur), 0)

        # Create an output image with the pixel values from the original sample image at the indexes where the mask has 
        # value 1 and replace the other pixel values (where mask has zero) with the blurred background.
        output_image = np.where(binary_mask_3, image, blurred_image)
    
        
    else:
        # Display the error message.
        print('Invalid Method')
        
        # Return
        return
    
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22, 22])
        plt.subplot(121)
        plt.imshow(image[:, :, ::-1])
        plt.title("Original Image")
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')
        plt.show()
    else:
        # Return the output image and the binary mask.
        # Also convert all the 1s in the mask into 255 and the 0s will remain the same.
        # The mask is returned in case you want to troubleshoot.
        return output_image, (binary_mask_3 * 255).astype('uint8')

# Carregar a imagem
image = cv2.imread("imagem3.png")
if image is None:
    print("Erro: A imagem não foi encontrada.")
    exit()

# Modificar o fundo da imagem
modifyBackground(image, method='blurBackground')