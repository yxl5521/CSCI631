import numpy as np
import pandas as pd
from matplotlib import patches

from functions import overlapScore

import torch

from cnn_model import cnn_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# load test data
# implement your code here
def test_model():
    testing = torch.tensor(
        pd.read_csv("./Dataset/testData.csv", header=None).to_numpy(), dtype=torch.float)
    ground_test = torch.tensor(
        pd.read_csv("./Dataset/newground-truth-test.csv", header=None).to_numpy(), dtype=torch.float)

    # load model.pth and test model

    model = cnn_model()
    model.eval()
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))

    # reshape your text data and feed into your model
    # implement your code here
    inputs, labels = testing.view(200, 1, 100, 100), ground_test.view(200, 4)
    outputs = model(inputs)

    # use overlapscore function to calculate the average score
    # implement your code here
    avgScore, scores = overlapScore(ground_test.detach().numpy(), outputs.detach().numpy())
    return scores, inputs, labels, outputs


if __name__ == '__main__':
    scores, inputs, labels, outputs = test_model()
    # save your output in a csv file in Result directory and draw an example with bounding box
    # implement your code here
    pd.DataFrame(np.array(scores)).to_csv("./Results/result.csv")
    print(inputs[0])
    print(scores[0])
    print(outputs[0])
    # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 5)
    plt.figure(figsize=(15, 10))
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(inputs[0][0], origin='lower')
    # Create a Rectangle patch
    output_1 = outputs[0]
    rect_pred = patches.Rectangle((output_1[0], output_1[1]), output_1[2], output_1[3], linewidth=1, edgecolor='r',
                                  facecolor='none')

    label_1 = labels[0]
    rect_truth = patches.Rectangle((label_1[0], label_1[1]), label_1[2], label_1[3], linewidth=1, edgecolor='r',
                                   facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect_truth)
    # ax.add_patch(rect_pred)
    plt.title('Image with Bounding Box')
    # plt.axis("off")
    plt.gray()
    plt.show()
