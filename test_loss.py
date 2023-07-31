import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(sci_mode=False)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x = self.fc2(x1)
        return x, x1


def CORAL(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = torch.matmul(torch.transpose(xm, 0, 1), xm)

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = torch.matmul(torch.transpose(xmt, 0, 1), xmt)

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss/(4*d*d)
    # import ipdb; ipdb.set_trace()

    return loss

#Let's go back to this
def CORAL_multi(source, target):
    channel_size = source.shape[1]
    #shape = [batch, channel, side, side] (assuming it's square)
    
    #Source
    #  Mean-center
    xm = torch.mean(source, dim=0, keepdim=True) - source 
    #Reduce dimension for each channel to only have 1 mean value
    #In the original implementation, the size of feature is only [batch, channel]
    #In testing this
    # print(xm, xm.shape)
    xm_2dim = torch.sum(xm, dim = (2,3)) 
    # print(xm_2dim, xm_2dim.shape)
    xc = torch.matmul(torch.transpose(xm_2dim, 0, 1), xm_2dim)
    

    #  Target
    xmt = torch.mean(target, dim=0, keepdim=True) - source
    xmt_2dim = torch.sum(xmt, dim = (2, 3))
    xct = torch.matmul(torch.transpose(xmt_2dim, 0, 1), xmt_2dim)

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss/(4*channel_size*channel_size)
    # print(channel_size)
    return loss/10000

def cmd(src_embed, tgt_embed, n_moments=5):
    if torch.mean(torch.abs(src_embed) + torch.abs(tgt_embed)) <= 1e-7:
        print("Warning: feature representations tend towards zero. "
              "Consider decreasing 'da_lambda' or using lambda schedule.")

    src_mean = src_embed.mean(dim=0)
    tgt_mean = tgt_embed.mean(dim=0)

    src_centered = src_embed - src_mean
    tgt_centered = tgt_embed - tgt_mean

    first_moment = l2diff(src_mean, tgt_mean)  # start with first moment

    moments_diff_sum = first_moment
    for k in range(2, n_moments + 1):
        moments_diff_sum = moments_diff_sum + moment_diff(src_centered, tgt_centered, k)

    return moments_diff_sum


def l2diff(src, tgt):
    """
    standard euclidean norm. small number added to increase numerical stability.
    """
    return torch.sqrt(torch.sum((src - tgt) ** 2) + 1e-8)


def moment_diff(src, tgt, moment):
    """
    difference between moments
    """
    ss1 = (src ** moment).mean(0)
    ss2 = (tgt ** moment).mean(0)
    return l2diff(ss1, ss2)

class CMD(object):
    #Inspired by https://gist.github.com/yusuke0519/724aa68fc431afadb0cc7280168da17b
    def __init__(self, n_moments=5):
        self.n_moments = n_moments

    def __call__(self, x1, x2):
        mx1 = torch.mean(x1, dim=0)
        mx2 = torch.mean(x2, dim=0)
        sx1 = x1 - mx1 
        sx2 = x2 - mx2

        scale_by_element_per_batch_first_moment = torch.prod(torch.tensor(mx1.shape[1:]))
        scale_by_element_per_batch_subseq_moment = torch.prod(torch.tensor(sx1.shape[1:]))

        dm = l2diff(mx1, mx2)/scale_by_element_per_batch_first_moment
        scms = dm

        for i in range(self.n_moments-1):
            # moment diff of centralized samples
            scms += moment_diff(sx1, sx2, i+2)/scale_by_element_per_batch_subseq_moment
        return scms


if __name__ == "__main__":
    x = torch.randn(10, 10)
    y = torch.randn(10, 10)
    model = MyModel()
    model2 = MyModel()
    model2.load_state_dict(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-0)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=1e-0)
    criterion = CMD()
    
    # for epoch in range(9):
    #     optimizer.zero_grad()
    #     output, aux = model(x)
    #     loss = criterion(output, y)
    #     print(loss)
    #     loss = loss + (aux**2).mean()
    #     loss.backward()
    #     optimizer.step()

    #     optimizer2.zero_grad()
    #     output2, _ = model2(x)
    #     loss2 = criterion(output2, y)
    #     print(loss2)
    #     loss2.backward()
    #     optimizer2.step()
        # print((model2.fc1.weight.detach() == model.fc1.weight.detach()).all())
        # print((model2.fc2.weight.detach() == model.fc2.weight.detach()).all())

    x = torch.randn(3, 10)
    y = torch.randn(3, 10)
    print(criterion(x,y))
    print(cmd_loss(x, y))

    # da_feature_maps = [torch.randn(2, 512, 80, 80) for x in range(4)]
    # for feature_maps in da_feature_maps:
    #     half_batch = feature_maps.shape[0]//2
    #     source_feature_maps = feature_maps[:half_batch,...]
    #     target_feature_maps = feature_maps[half_batch:,]
    #     print(CORAL_multi(source_feature_maps, target_feature_maps))

        # print(source_feature_maps.shape, target_feature_maps.shape)
    x = torch.randn(2, 512, 80, 80)
    y = torch.randn(2, 512, 80, 80)
    print(criterion(x, y))
    print(cmd_loss(x, y))


