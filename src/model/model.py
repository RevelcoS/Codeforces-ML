import sys

import torch
from torch import nn

import numpy as np

from tqdm import tqdm

class Model:

    def __init__(self):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.devide = torch.device(self.device)

        self.model = nn.Sequential(
            
            #nn.Conv2D(1, 10, (1, 10)),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            #nn.Softmax(dim=0),

        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=1e-3)

        self.epochs = 20

    def train(self, train_dataloader, test_dataloader):

        for epoch in range(self.epochs):

            print(f"Epoch = {epoch + 1}/{self.epochs}")
            self.model.train()

            progress = tqdm(train_dataloader, file=sys.stdout)

            for batch, (X, y) in enumerate(progress):
 
                pred = self.model(X)
                y = torch.unsqueeze(y, 1)

                loss = self.criterion(pred, y)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if (batch + 1) % 50 == 0:
                    loss = loss.item()
                    error = np.sqrt(loss) * 4000 + 800
                    progress.clear()
                    print(f"train_error: {error:.5f}")
                    progress.display()

            self.test(test_dataloader)

    def test(self, dataloader):

        self.model.eval()
        n_batches = len(dataloader)
        loss = 0

        with torch.no_grad():
            for (X, y) in dataloader:
                pred = self.model(X)
                y = torch.unsqueeze(y, 1)
                loss += self.criterion(pred, y).item()

        loss /= n_batches
        error = np.sqrt(loss) * 4000 + 800
        print(f"test_error: {error:.2f}")

    def predict(self, X):
        return self.model(X)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))

    def __repr__(self):
        return str(self.model)
