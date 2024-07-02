import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.uniform import Uniform
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Crear datos de entrenamiento
inputs = torch.tensor([[1., 0., 0., 0.],
                       [0., 1., 0., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., 1.]])

labels = torch.tensor([[0., 1., 0., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., 1.],
                       [0., 1., 0., 0.]])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset, batch_size=1)

class WordEmbeddingFromScratch(pl.LightningModule):

    def __init__(self):
        super().__init__()
        pl.seed_everything(seed=42)
        
        min_value, max_value = -0.5, 0.5
        
        self.input_w = nn.ParameterList([nn.Parameter(Uniform(min_value, max_value).sample([1])) for _ in range(8)])
        self.output_w = nn.ParameterList([nn.Parameter(Uniform(min_value, max_value).sample([1])) for _ in range(8)])
        
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, input): 
        input = input[0]
        inputs_to_top_hidden = sum(input[i] * self.input_w[2 * i] for i in range(4))
        inputs_to_bottom_hidden = sum(input[i] * self.input_w[2 * i + 1] for i in range(4))
        
        output = [inputs_to_top_hidden * self.output_w[2 * i] + inputs_to_bottom_hidden * self.output_w[2 * i + 1] for i in range(4)]
        return torch.stack(output)
        
    def configure_optimizers(self): 
        return Adam(self.parameters(), lr=0.1)
    
    def training_step(self, batch, batch_idx): 
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = self.loss(output_i.unsqueeze(0), label_i.unsqueeze(0))
        return loss
        
modelFromScratch = WordEmbeddingFromScratch()

# Mostrar parámetros iniciales
print("Before optimization, the parameters are...")
for name, param in modelFromScratch.named_parameters():
    print(name, torch.round(param.data, decimals=2))

# Visualización inicial de los parámetros
data = {
    "w1": [modelFromScratch.input_w[i].item() for i in range(0, 8, 2)],
    "w2": [modelFromScratch.input_w[i].item() for i in range(1, 8, 2)],
    "token": ["Troll2", "is", "great", "Gymkata"],
    "input": ["input1", "input2", "input3", "input4"]
}
df = pd.DataFrame(data)
sns.scatterplot(data=df, x="w1", y="w2")

for i, row in df.iterrows():
    plt.text(row["w1"], row["w2"], row["token"], horizontalalignment='left', size='medium', color='black', weight='semibold')
plt.show()

# Entrenamiento del modelo
trainer = pl.Trainer(max_epochs=100)
trainer.fit(modelFromScratch, train_dataloaders=dataloader)

# Mostrar parámetros después de la optimización
print("After optimization, the parameters are...")
for name, param in modelFromScratch.named_parameters():
    print(name, torch.round(param.data, decimals=2))

# Visualización de los parámetros optimizados
data = {
    "w1": [modelFromScratch.input_w[i].item() for i in range(0, 8, 2)],
    "w2": [modelFromScratch.input_w[i].item() for i in range(1, 8, 2)],
    "token": ["Troll2", "is", "great", "Gymkata"],
    "input": ["input1", "input2", "input3", "input4"]
}
df = pd.DataFrame(data)
sns.scatterplot(data=df, x="w1", y="w2")

for i, row in df.iterrows():
    plt.text(row["w1"], row["w2"], row["token"], horizontalalignment='left', size='medium', color='black', weight='semibold')
plt.show()

# Prueba del modelo
softmax = nn.Softmax(dim=0)

for input_tensor in inputs:
    output = modelFromScratch(input_tensor.unsqueeze(0))
    print(torch.round(softmax(output), decimals=2))
