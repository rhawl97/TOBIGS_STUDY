from loader import *


# Test the Model
correct = 0
total = 0
for i, data in enumerate(testloader):
    inputs, labels = data
#     images = images.view(-1, 28*28)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.shape[0]
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 2500 test images: %d %%' % (100 * correct / total))
