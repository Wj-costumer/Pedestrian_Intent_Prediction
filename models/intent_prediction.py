import torch
import torch.nn as nn
from .resnet import generate_model

class IntentPrediction(nn.Module):
    def __init__(self, model_opts, train_opts):
        super().__init__()
        self.model_opts = model_opts
        self.train_opts = train_opts
        self.dist_gru = nn.GRU(input_size=self.model_opts['input_size'], hidden_size=self.model_opts['hidden_size'], num_layers=self.model_opts['num_layers'], batch_first=True)
        self.resnet = generate_model(model_depth=self.model_opts['model_depth'],
                                     n_classes=self.model_opts['resnet_classes'],
                                     n_input_channels=self.model_opts['n_input_channels'],
                                     shortcut_type=self.model_opts['resnet_shortcut'],
                                     conv1_t_size=self.model_opts['conv1_t_size'],
                                     conv1_t_stride=self.model_opts['conv1_t_stride'],
                                     no_max_pool=self.model_opts['no_max_pool'],
                                     widen_factor=self.model_opts['resnet_widen_factor']
                                    )
        pretrain_model = torch.load(self.model_opts['pretrain_path'], map_location='cpu')
        self.resnet.load_state_dict(pretrain_model['state_dict'])
        self.output_layer = nn.Linear(self.model_opts['hidden_size'], self.model_opts['n_classes'])
        
    def forward(self, images, dist):
        dist = dist.to(torch.float32)
        images = images.transpose(1, 2) # torch.Size([32, 3, 16, 112, 112])
        dist_feats, _ = self.dist_gru(dist, torch.randn(self.model_opts['num_layers'], dist.size(0), self.model_opts['hidden_size']).to(dist.device)) # bs * hs * hidden_size
        action_feats = self.resnet(images) # torch.Size([32, 512])
        output = self.output_layer(dist_feats[:, -1, :] + action_feats)
        return output
        
    def train(self, dataloader):
        epoch = self.train_opts['epoch']
        lr = self.train_opts['lr']
        weight_decay = self.train_opts['weight_decay']
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        for i in range(epoch):
            for images, dist, labels in dataloader:
                optimizer.zero_grad()
                output = self.forward(images, dist)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                if (i+1) % 5 == 0:
                    print('Epoch [{}/{}], Loss: {:.4f}'.format(i+1, epoch, loss.item()))
                
    
    def eval(self, dataloader):
        pass