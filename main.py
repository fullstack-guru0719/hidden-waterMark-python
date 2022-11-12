#  Y  = R *  0.29900 + G *  0.58700 + B *  0.11400 
#  Cb = R * -0.16874 + G * -0.33126 + B *  0.50000 + 128 
#  Cr = R *  0.50000 + G * -0.41869 + B * -0.08131 + 128 

import numpy as np
# import matplotlib.pyplot as plt
import cv2
# from imwatermark import WatermarkEncoder, WatermarkDecoder
import math
import argparse

def bgr2ycbcr(img):
    Y = 0.29900 * img[:,:,2] + 0.58700 * img[:,:,1] + 0.11400 * img[:,:,0]
    Cb = -0.16874 * img[:,:,2] + -0.33126 * img[:,:,1] + 0.50000 * img[:,:,0] + 128
    Cr = 0.50000 * img[:,:,2] + -0.41869 * img[:,:,1] + -0.08131 * img[:,:,0] + 128

    Y = Y.astype('int')
    Cb = Cb.astype('int')
    Cr = Cr.astype('int')

    return Y, Cb, Cr

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype('float')
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

def create_wm_img(w, h, wm_text = 'Do not copy this!'):
    wm_img = np.zeros((h, w))

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = 1
    fontScale = 1.5
    org = (0, 100)
    thickness = 5

    size, _ = cv2.getTextSize(wm_text, font, fontScale, thickness)

    wm_w, wm_h = size
    space = 20
    for i in range(w // wm_w + 1):
        for j in range(h // wm_h + 1):
            org = ((j % 2) * space + (wm_w + space) * i, (wm_h + space) * j)
            wm_img = cv2.putText(wm_img, wm_text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    return wm_img, wm_w, wm_h

def calc_diff(img):
    return np.std(img).astype('int')

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def encode_img(img, wm):
    w, h = img.shape[1], img.shape[0]
    wm_img, wm_w, wm_h = create_wm_img(w, h, wm)

    Y, Cb, Cr = bgr2ycbcr(img)

    # Y, Cb, Cr = img[:,:,2], img[:,:,1], img[:,:,0]

    # plt.imshow(Y, cmap='gray')
    # plt.show()

    diff_size = 5
    diff = 0
    for i in range(h):
        for j in range(1, w):
            if i + diff_size < h and j + diff_size < w and i - diff_size > -1 and j - diff_size > -1:
                diff = calc_diff(Y[i-diff_size:i+diff_size, j-diff_size:j+diff_size])
                
                depth = (sigmoid(diff) - 0.5) * 15 + 1
                if wm_img[i, j] > 0:
                    Y[i, j] = Y[i, j] + (depth / 2 - 1) / 2 + depth / 2 - Y[i, j] % depth
                else:
                    Y[i, j] = Y[i, j] + (depth / 2 - 1) / 2 - Y[i, j] % depth   

    Y = np.expand_dims(Y, axis=2)
    Cr = np.expand_dims(Cr, axis=2)
    Cb = np.expand_dims(Cb, axis=2)

    YCbCr = np.concatenate((Y, Cb, Cr), axis=2)

    rgb = ycbcr2rgb(YCbCr)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # encoder = WatermarkEncoder()
    # encoder.set_watermark('bytes', wm.encode('utf-8'))
    # bgr_encoded = encoder.encode(bgr, 'dwtDctSvd')

    return bgr

def decode_img(img, wm_len):
    w, h = img.shape[1], img.shape[0]

    wm_text = []
    # if wm_len == -1:
    #     for i in range(30):
    #         try:
    #             decoder = WatermarkDecoder('bytes', i * 8)
    #             watermark = decoder.decode(img, 'dwtDctSvd')
    #             wm_text.append(watermark.decode('utf-8'))
    #         except:
    #             pass
    # else:
    #     try:
    #         decoder = WatermarkDecoder('bytes', wm_len * 8)
    #         watermark = decoder.decode(img, 'dwtDctSvd')
    #         wm_text.append(watermark.decode('utf-8'))
    #     except:
    #         pass

    Y, _, _ = bgr2ycbcr(img)

    wm_img = np.zeros((h, w))

    diff = 0
    diff_size = 3
    for i in range(h):
        for j in range(1, w):
            if i + diff_size < h and j + diff_size < w and i - diff_size > -1 and j - diff_size > -1:
                diff = calc_diff(Y[i-diff_size:i+diff_size, j-diff_size:j+diff_size])
                cnt = 0
                depth = (sigmoid(diff) - 0.5) * 15 + 1
                if Y[i, j] % depth > (depth / 2 - 1) / 2:
                    cnt += 1
                if cnt >= 1:
                    wm_img[i, j] = 255

    return wm_img, wm_text

def main(wm="WATERMARK", image="input.jpg", mode="encode", output="output.jpg"):
    if mode == "encode":
        img = cv2.imread(image)
        bgr = encode_img(img, wm)
        cv2.imwrite(output, bgr)
        print("OKAY")
        # img = cv2.imread(output)
        # cv2.imshow("encoded", img)
        # cv2.waitKey(0)

    else:
        bgr = cv2.imread(image)
        res_img, res_txt = decode_img(bgr, -1)
        cv2.imwrite(output, res_img)
        # print(res_txt)
        print("OKAY")
        # cv2.imshow("encoded", bgr)
        # cv2.imshow("watermark", res_img)
        # cv2.waitKey(0)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wm', type=str, default='WATERMARK')
    parser.add_argument('--image', type=str, default='input.jpg')
    parser.add_argument('--mode', type=str, default='encode')
    parser.add_argument('--output', type=str, default='output.jpg')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(**vars(opt))