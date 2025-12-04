import cv2
from skimage.metrics import structural_similarity as compare_ssim

def image_diff(imageA, imageB):
    # h, w = imageA.shape[:2]
    # new_h, new_w = h // 4, w // 4
    # imageA = cv2.resize(imageA, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # imageB = cv2.resize(imageB, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    return diff, score


image_1= cv2.imread('/Users/ibk5106/Desktop/IST_courses/TA/IST_597_003_AI_Private/rag_mcp/screenshots/screenshot_6435426512.png')
image_2 = cv2.imread('/Users/ibk5106/Desktop/IST_courses/TA/IST_597_003_AI_Private/rag_mcp/screenshots/screenshot_13127010512.png')

diff_img, diff_score = image_diff(image_1, image_2)

print("SSIM: {}".format(diff_score))