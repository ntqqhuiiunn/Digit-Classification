import torch, random, numpy, cv2, os 
from model import DigitsModel
from dataset import LabelCoder
def preprocess(image : numpy.ndarray):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[0], image.shape[1]
    if height != width:
        img = numpy.zeros((height, height), dtype= numpy.uint8)
        left_border = int((height - width) / 2)
        for h in range(height):
            for w in range(width):
                img[h][w + left_border] = image[h][w]
        return img
    else:
        return image
def detect(image : numpy.ndarray, model):
    # Input image must be a gray image with number of channels equals to 1.
    image = cv2.resize(image, (64, 64))

    model.eval()
    with torch.no_grad():
        img = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        img = torch.autograd.Variable(img)
        preds = model(img)
        _, predicted = torch.max(preds.data, 1)
        labelDecoder = LabelCoder("./data")
        predicted_result = labelDecoder.decode(predicted)
        return predicted_result[0]
if __name__ == "__main__":
    digit = random.choice(os.listdir("./data"))
    image_name = random.choice(os.listdir("./data/" + digit))
    cv2_image = cv2.imread("./data/" + digit + "/" + image_name)
    cv2.imshow("image: ", cv2_image)
    model = DigitsModel()
    model.load_state_dict(torch.load('./result/weights.ckpt'))

    processed = preprocess(cv2_image)
    res = detect(processed, model)
    print(res)
    cv2.waitKey(50000)
