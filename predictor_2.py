import cv2
import numpy as np
import matplotlib.pyplot as plt


def accessPixel(img):
    height = img.shape[0]
    width = img.shape[1]
    for i in range(height):
        for j in range(width):
            img[i][j] = 255 - img[i][j]
    return img


def accessBinary(img, threshold=10):
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    thresh, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    img = accessPixel(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img


def extractPeek(array_vals, min_vals=10, min_rect=20):
    extrackPoints = []
    startPoint = None
    endPoint = None
    for i, point in enumerate(array_vals):
        if point > min_vals and startPoint == None:
            startPoint = i
        elif point < min_vals and startPoint != None:
            endPoint = i

        if startPoint != None and endPoint != None:
            extrackPoints.append((startPoint, endPoint))
            startPoint = None
            endPoint = None

    for point in extrackPoints:
        if point[1] - point[0] < min_rect:
            extrackPoints.remove(point)
    return extrackPoints


def findBorderHistogram(path):
    borders = []
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = accessBinary(img)
    hori_vals = np.sum(img, axis=1)
    hori_points = extractPeek(hori_vals)
    for hori_point in hori_points:
        extractImg = img[hori_point[0]:hori_point[1], :]
        vec_vals = np.sum(extractImg, axis=0)
        vec_points = extractPeek(vec_vals, min_rect=0)
        for vect_point in vec_points:
            border = [(vect_point[0], hori_point[0]), (vect_point[1], hori_point[1])]
            borders.append(border)
    return borders


def findBorderContours(path, maxArea=1):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    threshold = 100
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = accessBinary(img)
    img = cv2.Canny(img, threshold, threshold * 2)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    borders = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > maxArea:
            border = [(x, y), (x + w, y + h)]
            borders.append(border)
    return borders


def showResults(path, borders, results=None):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = accessBinary(img)
    print(img.shape)
    for i, border in enumerate(borders):
        cv2.rectangle(img, border[0], border[1], (0, 0, 255))
        if results:
            cv2.putText(img, str(results[i]), border[0], cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
        # cv2.circle(img, border[0], 1, (0, 255, 0), 0)
    plt.imshow(img)
    plt.imsave('saved_figure.png', img)


def transMNIST(path, borders, size=(28, 28)):
    imgData = np.zeros((len(borders), size[0], size[0], 3, 1), dtype='uint8')

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold = 100
    img = cv2.Canny(img, threshold, threshold * 2)
    img = accessBinary(img)

    for i, border in enumerate(borders):
        borderImg = img[border[0][1]:border[1][1], border[0][0]:border[1][0]]
        extendPiexl = (max(borderImg.shape) - min(borderImg.shape)) // 2
        targetImg = cv2.copyMakeBorder(borderImg, 7, 7, extendPiexl + 7, extendPiexl + 7, cv2.BORDER_CONSTANT)
        targetImg = cv2.resize(targetImg, size)
        targetImg = np.expand_dims(targetImg, axis=-1)
        imgData[i] = targetImg
    return imgData


def predict(modelpath, imgData):
    result_number = []
    return result_number


if __name__ == '__main__':
    path = 'captcha/1624173627163216600.png'
    model = 'model.h5'
    borders = findBorderContours(path)
    imgData = transMNIST(path, borders)
    # results = predict(model, imgData)
    showResults(path, borders, range(len(borders)))
    # plt.imshow(borders)
    # plt.imsave('saved_figure.png', borders)
    print(borders)
