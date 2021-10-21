from keras.models import load_model
import argparse
import cv2

path = "D:\MSCS\Research\image processing\keras-tutorial\images\panda.jpg"
#path = "D:\MSCS\Research\image processing\image2csv\catdog\dogs\dog.10.jpg"
img_size = 64
image = cv2.imread(path)
output = image.copy()

image = cv2.resize(image, (img_size, img_size))
image = image.astype("float") / 255.0
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

model = load_model("D:\MSCS\Research\image processing\keras-tutorial\SmallVGG.model")

preds = model.predict(image)
print(preds)

i = preds.argmax(axis=1)[0]
CATEGORIES = ["cat", "dog", "panda"]
num = 0
for i in range(len(CATEGORIES)):
    if (preds[0][i]*100) > num:
        num = i

text = "{}: {:.2f}%".format(CATEGORIES[num], preds[0][i] * 100)
print(text)

cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# show the output image
cv2.imshow("Image", output)
cv2.waitKey(0)