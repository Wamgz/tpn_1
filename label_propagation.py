import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim) # 需要是float才行，long不行
        self.fn = fn

    def forward(self, x, **kwargs):
        x = rearrange(x, 'b n e -> b e n')
        return self.fn(rearrange(self.norm(x), 'b e n -> b n e'), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            Rearrange('b n e -> b e n'),
            nn.BatchNorm1d(dim),
            Rearrange('b e n -> b n e'),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, embed_dim, class_embed_dim, num_patch, heads=8, dim_head=64, dropout=0., use_linear_v=True):
        super().__init__()
        inner_dim = dim_head * heads # 1024
        project_out = False
        self.embed_dim = embed_dim
        self.class_embed_dim = class_embed_dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.num_patch = num_patch
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qk = nn.Linear(embed_dim, inner_dim * 2, bias=False)

        if use_linear_v:
            self.to_v = nn.Linear(embed_dim + class_embed_dim, inner_dim + class_embed_dim, bias=False) # TODO 是否需要linear
        else:
            self.to_v = nn.Identity()


        self.to_out = nn.Sequential(
            nn.Linear(inner_dim + class_embed_dim, embed_dim + class_embed_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    ## TODO 尝试batch * num_patch做为一个大batch
    def forward(self, x):
        # x: (batch, num_patch, embed_dim + class_embed_dim) -> (batch * num_patch, embed_dim + class_embed_dim)
        # q, k -> x[:, :, :embed_dim]
        batch, num_patch, dim = x.shape
        x = rearrange(x, 'b n d -> (b n) d') # (batch * num_patch, embed_dim + class_embed_dim)
        qk = self.to_qk(x[:, :-self.class_embed_dim]).chunk(2, dim=-1) # tuple: ((batch, num_patch, inner_dim))
        v = self.to_v(x) # (batch * num_patch , inner_dim + class_embed_dim)
        q, k = qk
        # q, k = map(lambda t: rearrange(t, 'B (h d) -> h B d', h=self.heads), qk) # (num_head, batch * num_patch, head_dim)
        # v = rearrange(v, 'B (h d) -> h B d', h=self.heads) #  (num_head, batch * num_patch, head_dim)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # (batch * num_patch, batch * num_patch)

        attn = self.attend(dots) # q和k的相似度矩阵, attn: (batch * num_patch, batch * num_patch)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v) # attn矩阵乘v不是点乘（对v加权），v:(batch * num_patch, inner_dim + class_embed_dim)
        out = rearrange(out, '(b n) d -> b n d', b = batch, n = num_patch) # (batch, num_patch, inner_dim)

        return self.to_out(out) # TODO 分开过？


class Transformer(nn.Module):
    def __init__(self, embed_dim, class_embed_dim, depth, heads, dim_head, mlp_dim, num_patch, dropout=0., use_linear_v=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([ # 这里是先进行norm，再进行Attention和FFN
                PreNorm(embed_dim + class_embed_dim, Attention(embed_dim, class_embed_dim=class_embed_dim, num_patch=num_patch, heads=heads, dim_head=dim_head, dropout=dropout, use_linear_v=use_linear_v)),
                PreNorm(embed_dim + class_embed_dim, FeedForward(embed_dim + class_embed_dim, mlp_dim + class_embed_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=84, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # （1, 3, 224, 224） -> (1, 96, 56 ,56) -> (1, 96, 56 * 56) -> (1, 3136, 96)B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, out_dim, embed_dim, depth, heads, mlp_dim, class_embed_dim=5, total_class=100, cls_per_episode=5,
                 support_num=5, query_num=15, pool='cls', channels=1, dim_head=12, tsfm_dropout=0., emb_dropout=0., feature_only=False, pretrained=False, patch_norm=True, conv_patch_embedding=False,
                 use_avg_pool_out=False, use_dual_feature=False, use_linear_v=True):
        super().__init__()
        self.pretrained = pretrained

        image_height, image_width = pair(image_size) #
        patch_height, patch_width = pair(patch_size) # 32, 32
        self.num_support, self.num_query = support_num, query_num
        self.cls_per_episode = cls_per_episode
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.num_patch = (image_height // patch_height) * (image_width // patch_width) # 64
        patch_dim = channels * patch_height * patch_width # 3072
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.conv_patch_embedding = conv_patch_embedding
        if self.conv_patch_embedding:
            ## 卷积实现patch_embedding
            self.to_patch_embedding = PatchEmbed(
                img_size=image_size, patch_size=patch_size, in_chans=channels, embed_dim=embed_dim,
                norm_layer=nn.LayerNorm if patch_norm else None)
        else:
            ## MLP实现patch_embedding
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
                nn.Linear(patch_dim, embed_dim), # patch dim: 3072, dim: 1024
                nn.LayerNorm(embed_dim)
            )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patch, embed_dim))
        trunc_normal_(self.pos_embedding, std=.02)

        self.class_embed_dim = self.cls_per_episode

        # self.cls_token = nn.Parameter(torch.zeros(self.cls_per_episode + 1, self.class_embed_dim)) # patch维度的class_embed
        # torch.nn.init.orthogonal_(self.cls_token, gain=1)

        self.dropout = nn.Dropout(emb_dropout)
        # dim: 1024, depth: 6, heads: 16, dim_head: 64, mlp_dim: 2048, dropout: 0.1
        self.transformer = Transformer(embed_dim, class_embed_dim, depth, heads, dim_head, mlp_dim, self.num_patch, tsfm_dropout, use_linear_v=use_linear_v)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim)
        )
        self.use_avg_pool_out = use_avg_pool_out
        self.norm = nn.LayerNorm(embed_dim)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.out_head = nn.Sequential(
            nn.LayerNorm((self.num_patch + 1) * embed_dim),
            nn.Linear((self.num_patch + 1) * embed_dim, out_dim)
        )

        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.apply(self._init_weights)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.total_class = total_class
        self.use_dual_feature = use_dual_feature
        self.avg_pool_64 = nn.AdaptiveAvgPool1d(64)
    def forward(self, batch):
        '''
        :param imgs: (batch, C, H, W) -> (100, 3, 96, 96)
        :param labels: (batch, ) -> (100, )
        :return:
        '''
        ## patch embedding
        sup_imgs, query_imgs, sup_labels, query_labels = batch
        imgs, labels = torch.cat((sup_imgs, query_imgs), dim=0), torch.cat((sup_labels, query_labels), dim=0)
        x = self.to_patch_embedding(imgs) # (batch, num_patch, patch_size * patch_size) -> (100, 12 * 12, 64)
        if self.use_dual_feature:
            x_1 = self.to_patch_embedding(F.interpolate(imgs, [64, 64]))
            x_2 = self.to_patch_embedding(F.interpolate(imgs, [32, 32]))
            x = torch.cat((x, x_1, x_2), dim=1) # num_patch维度拼接
            x = self.avg_pool_64(x.transpose(1, 2)).transpose(1, 2)

        batch, num_patch, _ = x.shape
        x += self.pos_embedding[:, :num_patch] # (batch, num_patch, embed_dim)

        labels = self._map2ZeroStart(labels)
        labels_unique, _ = torch.sort(torch.unique(labels))

        ## 拆分support和query，加上对应的class_embedding
        support_idxs, query_idxs = self._support_query_data(labels)
        support_cls_token, query_cls_token = torch.nn.functional.one_hot(labels[support_idxs], self.cls_per_episode), torch.zeros(query_idxs.size(0), num_patch, self.cls_per_episode) # (num_support, class_per_episode)
        if torch.cuda.is_available():
            query_cls_token = query_cls_token.cuda()
        support_cls_tokens, query_cls_tokens = \
            support_cls_token.unsqueeze(1).repeat(1, self.num_patch, 1), query_cls_token # (num_support, num_patch, class_embed_dim)
        # support_x1, query_x1 = x[support_idxs], x[query_idxs]
        support_x, query_x = torch.cat((x[support_idxs], support_cls_tokens), dim=-1), torch.cat((x[query_idxs], query_cls_tokens), dim=-1) # patch维度拼接, (num_support, num_patch, embed_dim + embed_dim), (num_query, num_patch, embed_dim + embed_dim)
        x, labels = torch.cat((support_x, query_x), dim=0), torch.cat((labels[support_idxs], labels[query_idxs]))

        ## transformer
        x = self.dropout(x)
        x = self.transformer(x) # (batch, num_patch, embedding_dim + class_embed_dim)

        ## 取出class_embed进行loss计算，(support和query）都计算
        logits = self.avg_pool(x[:, :, -self.class_embed_dim:].transpose(1, 2)).transpose(1, 2).squeeze(1) # (batch, class_embed_dim)
        x_entropy = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            x_entropy = x_entropy.cuda()
        loss = x_entropy(logits, labels) # (batch, class_per_epi)

        y_hat = torch.argmax(logits[self.num_support * self.cls_per_episode:, :], 1)
        acc_val = y_hat.eq(labels[self.num_support * self.cls_per_episode:]).float().mean()

        return loss, acc_val


    def _support_query_data(self, labels):
        labels_unique, _ = torch.sort(torch.unique(labels))
        support_idxs = torch.stack(list(map(lambda c: labels.eq(c).nonzero()[:self.num_support], labels_unique))).view(-1)  # (class_per_episode * num_support)
        query_idxs = torch.stack(list(map(lambda c: labels.eq(c).nonzero()[self.num_support:], labels_unique))).view(-1)  # (class_per_episode * num_query)
        return support_idxs, query_idxs


    def _map2ZeroStart(self, labels):
        labels_unique, _ = torch.sort(torch.unique(labels))
        labels_index = torch.zeros(self.total_class)
        for idx, label in enumerate(labels_unique):
            labels_index[label] = idx
        for i in range(labels.size(0)):
            labels[i] = labels_index[labels[i]]
        return labels

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def trainable_params(self):
        if self.pretrained:
            return self.pretrained_model.head.parameters()
        return self.parameters()

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

    M = 1024 * 1024
    size = total_num / 4. / M
    print('参数量: %d\n模型大小: %.4fM' % (total_num, size))
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == '__main__':
    model = ViT(
            image_size=96,
            patch_size=8,
            out_dim=64,
            embed_dim=64,
            depth=4,
            heads=8,
            dim_head=8,
            mlp_dim=64,
            tsfm_dropout=0.0,
            emb_dropout=0.0,
            use_avg_pool_out=True,
            channels=3
        )

    support = torch.randn((25, 3, 96, 96))
    query = torch.randn((75, 3, 96, 96))
    imgs = torch.cat((support, query), 0)
    support_labels = torch.arange(5).view(1, -1).repeat(5, 1).view(-1) + 2
    query_labels = torch.arange(5).view(1, -1).repeat(15, 1).view(-1) + 2
    support_labels, query_labels = support_labels[torch.randperm(25)], query_labels[torch.randperm(75)]
    labels = torch.cat((support_labels, query_labels), 0)
    out = model(imgs, labels)

    print(out)
    # num_param = get_parameter_number(model)