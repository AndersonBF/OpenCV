import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# Construir a lookup table para aumentar o valor dos pixels
increase_table = np.clip(UnivariateSpline(x=[0, 64, 128, 255], y=[0, 75, 155, 255])(range(256)), 0, 255).astype(np.uint8)

# Construir a lookup table para diminuir o valor dos pixels
decrease_table = np.clip(UnivariateSpline(x=[0, 64, 128, 255], y=[0, 45, 95, 255])(range(256)), 0, 255).astype(np.uint8)

def applyWarm(image):
    blue_channel, green_channel, red_channel = cv2.split(image)
    # Aumenta o vermelho
    red_channel = cv2.LUT(red_channel, increase_table).astype(np.uint8)
    # Diminui o azul
    blue_channel = cv2.LUT(blue_channel, decrease_table).astype(np.uint8)
    # Mesclar os canais novamente
    output_image = cv2.merge((blue_channel, green_channel, red_channel))
    return output_image

def applyCold(image):
    blue_channel, green_channel, red_channel = cv2.split(image)
    # Aumenta o azul
    blue_channel = cv2.LUT(blue_channel, increase_table).astype(np.uint8)
    # Diminui o vermelho
    red_channel = cv2.LUT(red_channel, decrease_table).astype(np.uint8)
    # Mesclar os canais novamente
    output_image = cv2.merge((blue_channel, green_channel, red_channel))
    return output_image


midtone_contrast_increase = UnivariateSpline(x=[0, 25, 51, 76, 102, 128, 153, 178, 204, 229, 255],
                                                    y=[0, 13, 25, 51, 76, 128, 178,  204, 223, 239, 255])(range(256))
    
lowermids_increase = UnivariateSpline (x=[0, 16, 32, 48, 64, 80, 96, 111, 128, 143, 159, 175, 191, 207, 223, 239, 255], 
                                           y=[0, 18, 35, 64, 81, 99, 107, 112, 121, 143, 159, 175, 191, 207, 223, 239, 255]) (range(256))
    
uppermids_decrease = UnivariateSpline (x=[0, 16, 32, 48, 64, 80, 96, 111, 128, 143, 159, 175, 191, 207, 223, 239, 255], 
                                       y=[0, 16, 32, 48, 64, 80, 96, 111, 128, 140, 148, 160, 171, 187, 216, 236, 255]) (range(256))
print(f'primeiros 10 elementos midtone: \n {midtone_contrast_increase[:10]}\n')
print(f'primeiros 10 elementos lowermids: \n {lowermids_increase[:10]}\n')
print(f'primeiros 10 elementos uppermids: \n {uppermids_decrease[:10]}\n')

def applyGotham(image, display=True):
    blue_channel, green_channel, red_channel = cv2.split(image)
    # Aumenta o contraste dos tons médios
    red_channel = cv2.LUT(red_channel, midtone_contrast_increase).astype(np.uint8)

    blue_channel = cv2.LUT(blue_channel, lowermids_increase).astype(np.uint8)
    # Aumenta os tons mais escuros
    blue_channel = cv2.LUT(blue_channel, uppermids_decrease).astype(np.uint8)
    # Diminui os tons mais claros
    # Mesclar os canais novamente
    output_image = cv2.merge((blue_channel, green_channel, red_channel))
    return output_image

def grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    output_image = cv2.merge((gray, gray, gray)) 
    return output_image  

def applySepia(image, display=True):
    #converter a imagem em float
    image_float = np.array(image, dtype=np.float64)
    blue_channel, green_channel, red_channel = cv2.split(image)
    # Aplicar os filtros
    output_blue = (red_channel * 0.272) + (green_channel * 0.534) + (blue_channel * 0.131)
    output_green = (red_channel * 0.349) + (green_channel * 0.686) + (blue_channel * 0.168)
    output_red = (red_channel * 0.393) + (green_channel * 0.769) + (blue_channel * 0.189)
    # Mesclar os canais novamente
    output_image = cv2.merge((output_blue, output_green, output_red))

    #sepia para os 3 espaços de cor
    sepia_matrix =np.matrix([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    output_image[output_image > 255] = 255
    output_image = np.array(output_image, dtype=np.uint8)
    return output_image

# Carregar a imagem
image = cv2.imread("imagem3.png")
if image is None:
    print("Erro: A imagem não foi encontrada.")
    exit()

# Aplicar os filtros
warm_image = applyWarm(image)
cold_image = applyCold(image)
gotham_image = applyGotham(image)
gray_image = grayscale(image)
output_image = applySepia(image)

# Exibir as imagens na mesma janela
plt.figure(figsize=[12, 8])  # Ajustar o tamanho da janela
plt.subplot(161)
plt.imshow(image[:, :, ::-1])
plt.title('Imagem Original')
plt.axis('off')

plt.subplot(162)
plt.imshow(warm_image[:, :, ::-1])
plt.title('Filtro Quente')
plt.axis('off')

plt.subplot(163)
plt.imshow(cold_image[:, :, ::-1])
plt.title('Filtro Frio')
plt.axis('off')

plt.subplot(164)
plt.imshow(gotham_image[:, :, ::-1])
plt.title('Filtro Gotham')
plt.axis('off')

plt.subplot(165)
plt.imshow(gray_image[:, :, ::-1])
plt.title('Escala de Cinza')
plt.axis('off')

# Exibir a janela
plt.subplot(166)
plt.imshow(output_image[:, :, ::-1])
plt.title('Filtro Sepia')
plt.axis('off')


plt.show()