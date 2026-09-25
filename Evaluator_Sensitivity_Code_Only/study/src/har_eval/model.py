import torch
from torch import nn
from .io import stream
class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.conv=nn.Sequential(nn.Conv1d(9,16,5,padding=2),nn.ReLU(),nn.Conv1d(16,32,5,padding=2),nn.ReLU()); self.fc=nn.Linear(64,6)
    def forward(self,x):
        z=self.conv(x); return self.fc(torch.cat([z.mean(2),z.amax(2)],1))
def fit(state,x,y,seed,cfg):
    m=Model();m.load_state_dict(state);m.train()
    opt=torch.optim.SGD(m.parameters(),lr=cfg['learning_rate'],momentum=0,weight_decay=0)
    gen=torch.Generator().manual_seed(seed); order=torch.randperm(len(y),generator=gen); pos=0; total=0.; n=0
    for step in range(cfg['local_steps']):
        if pos>=len(y): order=torch.randperm(len(y),generator=gen);pos=0
        ids=order[pos:pos+cfg['batch_size']];pos+=len(ids)
        opt.zero_grad(); loss=nn.functional.cross_entropy(m(x[ids]),y[ids]);loss.backward();opt.step()
        total+=float(loss.detach())*len(ids);n+=len(ids)
    return {k:v.detach().clone() for k,v in m.state_dict().items()},dict(ce_sum=total,processed_examples=n,minibatch_steps=cfg['local_steps'])
def predict(state,x):
    m=Model();m.load_state_dict(state);m.eval()
    with torch.no_grad(): return torch.cat([m(b) for b in x.split(256)]).numpy()
