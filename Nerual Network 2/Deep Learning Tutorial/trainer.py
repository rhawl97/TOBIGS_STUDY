from loader import *
from models import model
from configs import src

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class Trainer():
    def __init__(self, src, trainloader, testloader, model):
        self.batch_size = src['batch_size']
        self.num_epochs = src['num_epochs']
        self.lr = src['learning_rate']

        self.model = model
        self.model.init_weight()
        self.train = trainloader
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #cuda가 있으면 gpu가 돌아감 

    def _train(self):
        model = self.model
        device = self.device
        model = model.to(device)  #gpu tensor적용 
        print('Current Mode is:', device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(),
                              self.lr,
                              momentum=0.9)
        best_loss = 10000

        for epoch in tqdm(range(self.num_epochs)):
            current_loss = 0.0
            recent_loss = 0
        #     model.train(True)

            for i, data in enumerate(self.train):
                # get the inputs
                inputs, labels = data
                inputs = inputs.to(device)  #tensor에다가는 전부 붙혀야 함 
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                step = i + 1
                current_loss += loss.item()

                if step % 1000 == 0 and step != 0:     # print every 1000 mini-batches
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %
                          (epoch + 1, self.num_epochs, step, len(self.train)//1000 * 1000, current_loss / 1000))
                    recent_loss = current_loss
                    current_loss = 0.0

            if best_loss > recent_loss:
                best_loss = recent_loss
                torch.save(model.state_dict(), 'model' + str(recent_loss)[:6], '.pkl')


if __name__ == '__main__':
    trainer = Trainer(src, trainloader, testloader, model)
    trainer._train()
