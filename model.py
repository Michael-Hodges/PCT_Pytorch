import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sample_and_group 
import argparse

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)   
        x = x.reshape(-1, d, s) 
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # B, D, N
        x = F.relu(self.bn2(self.conv2(x))) # B, D, N
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x

class pct_semantic(nn.Module):
    def __init__(self, args, output_channels=27):
        super(pct_semantic, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last(args,channels=256)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.lbrd1 = nn.Sequential(nn.Conv1d(2048, 256, kernel_size=1, bias=False), 
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(negative_slope=0.2),
                                    nn.Dropout(p=0.5))
        self.lbr1 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.linear_final = nn.Conv1d(256, output_channels,kernel_size=1, bias = False)

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        # b, d, n
#        print('input shape: {0}'.format(x.size()))
        x = F.relu(self.bn1(self.conv1(x)))
#        print('embedingshape1: {0}'.format(x.size()))
        # b, d, n
        x = F.relu(self.bn2(self.conv2(x)))
#        print('embedingshape2: {0}'.format(x.size()))
#        x = x.permute(0, 2, 1)
#        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=1, xyz=xyz, points=x)         
#        feature_0 = self.gather_local_0(new_feature)
#        print('feature0: {0}'.format(feature_0.size()))
#        feature = feature_0.permute(0, 2, 1)
#        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=1, xyz=new_xyz, points=feature) 
#        feature_1 = self.gather_local_1(new_feature) #replaced new_feature with feature_0 
#        print('feature1: {0}'.format(feature_1.size()))
        feature_1 = x
        x = self.pt_last(x)
#        print('ptlast1: {0}'.format(x.size()))
        x = torch.cat([x, feature_1], dim=1)
#        print('post concat: {0}'.format(x.size()))
        x = self.conv_fuse(x)   # Point Features 1024
#        print('point feature size: {0}'.format(x.size()))
        point_feature = x
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # GLobal Features
#        print('global feature size: {0}'.format(x.size()))
        global_feat = torch.unsqueeze(x, 2)
#        print('global feature unsqueeze: {0}'.format(global_feat.size()))
        global_feat = global_feat.expand((-1,-1, point_feature.size()[2]))#x.repeat((point_feature.size()[0],point_feature.size()[1],256))

#        print('global feat rep size: {0}'.format(global_feat.size()))
#        print('point feat size:{0}'.format(x.size()))
        # concat a repeat of global features and point features
        x = torch.cat([global_feat, point_feature], dim=1)
#        print('concat size: {0}'.format(x.size()))
        # conv 1D LBRD 256
        x = self.lbrd1(x)
#        print('lbrd1 size: {0}'.format(x.size()))
        x = self.lbr1(x)
#        print('lbr1 size: {0}'.format(x.size()))
        x = self.linear_final(x)
#        print('final size: {0}'.format(x.size()))
        # conv 1D LBR 256
        # conv 1D Linear Ns = 27 since k^3 is 27



#        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
#        x = self.dp1(x)
#        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
#        x = self.dp2(x)
#        x = self.linear3(x)

        return x

class pct_simple(nn.Module):
    def __init__(self, args, output_channels=40):
        super(pct_simple, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last(args,channels=256)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

#        self.lbrd1 = nn.Sequential(nn.Conv1d(2048, 256, kernel_size=1, bias=False), 
#                                    nn.BatchNorm1d(256),
#                                    nn.LeakyReLU(negative_slope=0.2),
#                                    nn.Dropout(p=0.5))
#        self.lbr1 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
#                                    nn.BatchNorm1d(256),
#                                    nn.LeakyReLU(negative_slope=0.2))
#        self.linear_final = nn.Conv1d(256, output_channels,kernel_size=1, bias = False)
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        # b, d, n
        x = F.relu(self.bn1(self.conv1(x)))
        # b, d, n
        x = F.relu(self.bn2(self.conv2(x)))
        feature_1 = x

        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

#        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
#        x = self.dp1(x)
#        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
#        x = self.dp2(x)
#        x = self.linear3(x)

        return x
class Pct(nn.Module):
    def __init__(self, args, output_channels=40):
        super(Pct, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last(args)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))


        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        # b, d, n
        x = F.relu(self.bn1(self.conv1(x)))
        # b, d, n
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)

        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x

class Pct_semantic(nn.Module):
    def __init__(self, args, output_channels=40):
        super(Pct_semantic, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last(args)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.lbrd1 = nn.Sequential(nn.Conv1d(2048, 256, kernel_size=1, bias=False), 
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(negative_slope=0.2),
                                    nn.Dropout(p=0.5))
        self.lbr1 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.linear_final = nn.Conv1d(256, output_channels,kernel_size=1, bias = False)

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        # b, d, n
        x = F.relu(self.bn1(self.conv1(x)))
        # b, d, n
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=2048, radius=0.15, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=2048, radius=0.2, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)

        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        point_feature = x
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # GLobal Features
#        print('global feature size: {0}'.format(x.size()))
        global_feat = torch.unsqueeze(x, 2)
#        print('global feature unsqueeze: {0}'.format(global_feat.size()))
        global_feat = global_feat.expand((-1,-1, point_feature.size()[2]))#x.repeat((point_feature.size()[0],point_feature.size()[1],256))

        # concat a repeat of global features and point features
        x = torch.cat([global_feat, point_feature], dim=1)
#        print('concat size: {0}'.format(x.size()))
        # conv 1D LBRD 256
        x = self.lbrd1(x)
#        print('lbrd1 size: {0}'.format(x.size()))
        x = self.lbr1(x)
#        print('lbr1 size: {0}'.format(x.size()))
        x = self.linear_final(x)

        return x

class Point_Transformer_Last(nn.Module):
    def __init__(self, args, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    args = parser.parse_args()
    test_sem = pct_semantic(args).cuda()
    x = torch.rand((1,3,2048), dtype=torch.float32).cuda()
    test_sem.forward(x)


