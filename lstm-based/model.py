from utils import *


def compute_loss(predictions, labels, subgroups, power=1.0,
            score_function=nn.BCEWithLogitsLoss(reduction='none')):
    """
    predictions: (N, )
    labels: (N, )
    subgroups: (N, 9)
    """
    subgroup_positive_mask = subgroups & (labels.unsqueeze(-1) >= 0.5)
    subgroup_negative_mask = subgroups & ~(labels.unsqueeze(-1) >= 0.5)
    background_positive_mask = ~subgroups & (labels.unsqueeze(-1) >= 0.5)
    background_negative_mask = ~subgroups & ~(labels.unsqueeze(-1) >= 0.5)

    bpsn_mask = (background_positive_mask | subgroup_negative_mask).float()
    bnsp_mask = (background_negative_mask | subgroup_positive_mask).float()
    subgroups = subgroups.float()
    predictions = predictions.float()
    labels = labels.float()

    bce = score_function(predictions, labels)   # (N, )
    sb = (bce.unsqueeze(-1) * subgroups).sum(0).div(subgroups.sum(0).clamp(1.))\
            .mean()
#             .pow(power).mean().pow(1/power)
    bpsn = (bce.unsqueeze(-1) * bpsn_mask).sum(0).div(bpsn_mask.sum(0).clamp(1.))\
            .mean()
#             .pow(power).mean().pow(1/power)
    bnsp = (bce.unsqueeze(-1) * bnsp_mask).sum(0).div(bnsp_mask.sum(0).clamp(1.))\
            .mean()
#             .pow(power).mean().pow(1/power)
    return (bce.mean() + sb + bpsn + bnsp) / 4


class UnbiasLoss(nn.Module):
    def __init__(self, main_loss_weight=1.0):
        super(UnbiasLoss, self).__init__()
        self.alpha = main_loss_weight
    def forward(self, pred_scores, labels):
        subgs = (labels[:, -len(identity_columns):] >= 0.5)
        main_loss = compute_loss(pred_scores[:,0], labels[:,0], subgs)
        aux_loss = nn.BCEWithLogitsLoss()(pred_scores[:,1:-len(identity_columns)], labels[:,1:-len(identity_columns)])
        return self.alpha * main_loss + aux_loss


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Attention(nn.Module):
    def __init__(self, feature_dim, dropout=0.1, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.feature_dim = feature_dim
        self.proj_to_hid = nn.Linear(feature_dim, feature_dim)
        self.proj_to_alpha = nn.Linear(feature_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq):   # seq: (N, T, D)
        mid = torch.tanh(self.proj_to_hid(seq))   # (N, T, D)
        eij = self.proj_to_alpha(mid).squeeze(-1)   # (N, T)
        alpha = F.softmax(eij, dim=-1)   # (N, T)
        context = torch.sum(self.dropout(alpha).unsqueeze(-1) * seq, 1)   # (N, D)
#         context = self.dropout(alpha).unsqueeze(-1) * seq   # (N, T, D)

        return context, alpha


class AttentionEmb(nn.Module):
    # Attention on k embeddings

    def __init__(self, embed_dim, **kwargs):
        super(AttentionEmb, self).__init__(**kwargs)
        self.proj_to_alpha = nn.Linear(embed_dim, 1)
        nn.init.xavier_uniform_(self.proj_to_alpha.weight)
        self.proj_to_alpha.bias.data.fill_(0.)

    def forward(self, embs):   # embs: (N, T, k, M)
        eij = self.proj_to_alpha(embs).squeeze(-1)   # (N, T, k)
        alpha = F.softmax(eij, dim=-1)   # (N, T, k)
        w_emb = torch.sum(alpha.unsqueeze(-1) * embs, 2)   # (N, T, M)
        return w_emb


class LayerNorm(nn.Module):
    "Construct a layernorm module."
    def __init__(self, feature_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(feature_dim))
        self.b_2 = nn.Parameter(torch.zeros(feature_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, dim, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(-1)    # (N, T, M, 1)
        x = x.transpose(1,2)   # (N, M, T, 1)
        x = super(SpatialDropout, self).forward(x)  # (N, M, T, 1), some features are masked
        x = x.squeeze(-1)     # (N, M, T)
        x = x.transpose(1,2)   # (N, T, M)
        return x


class EmbeddingLayer(nn.Module):

    def __init__(self, vocab_size, embed_dim, embed_matrix):
        super(EmbeddingLayer, self).__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.emb.weight = nn.Parameter(torch.tensor(embed_matrix, dtype=torch.float32))
        self.emb_dropout = SpatialDropout(0.35)

    def forward(self, seq):
        emb = self.emb(seq)
        emb = self.emb_dropout(emb)
        return emb


class RecurrentNet(nn.Module):

    def __init__(self, embed_dim, hidden_dim):
        super(RecurrentNet, self).__init__()
        # Init layers
        self.lstm = nn.LSTM(embed_dim, 120, bidirectional=True, batch_first=True)
#         self.lstm2 = nn.LSTM(hidden_dim*2, hidden_dim, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(120*2, 60, bidirectional=True, batch_first=True)
#         self.sublayers = clones(SublayerConnection(hidden_dim*2, 0.1), 2)

        for mod in (self.lstm, self.gru):
            for name, param in mod.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param)

    def forward(self, seq):
        o_lstm, _ = self.lstm(seq)
#         o_lstm2 = self.sublayers[0](o_lstm, lambda x: self.lstm2(x)[0])
        o_gru, h_gru = self.gru(o_lstm)
        return o_gru, h_gru


class CommentClassifier(nn.Module):

    def __init__(self, hidden_dim, output_dim):
        super(CommentClassifier, self).__init__()
#         self.attn = Attention(hidden_dim*2)
        self.dropout = nn.Dropout(0.1)
        self.fc_out = nn.Linear(hidden_dim*4, output_dim)

    def forward(self, seq, hidden):
#         o_atten, _ = self.attn(seq)
        avg_pool = torch.mean(seq, 1)
        max_pool, _ = torch.max(seq, 1)
        h_concat = torch.cat((avg_pool, max_pool), 1)
        out = self.fc_out(self.dropout(h_concat))
        return out


class JigsawNet(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, embed_matrix):
        super(JigsawNet, self).__init__()
        # Init layers
        self.emb_layer = EmbeddingLayer(vocab_size, embed_dim, embed_matrix)
        self.rnns = RecurrentNet(embed_dim, hidden_dim)
        self.classifier = CommentClassifier(hidden_dim, 17)

    def forward(self, seq):
        emb = self.emb_layer(seq)
        o_rnn = self.rnns(emb)
        out = self.classifier(o_rnn)

        return out


class WeightEMA(object):
    def __init__(self, model, mu=0.95, sample_rate=1):
        # self.ema_model = copy.deepcopy(model)
        self.mu = mu
        self.sample_rate = sample_rate
        self.sample_cnt = sample_rate
        self.weight_copy = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.weight_copy[name] = (1 - mu) * param.data

    def _update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1 - self.mu) * param.data + self.mu * self.weight_copy[name]
                self.weight_copy[name] = new_average.clone()

    def set_weights(self, ema_model):
        for name, param in ema_model.named_parameters():
            if param.requires_grad:
                param.data = self.weight_copy[name]

    def on_batch_end(self, model):
        self.sample_cnt -= 1
        if self.sample_cnt == 0:
            self._update(model)
            self.sample_cnt = self.sample_rate


def model_test():
    x = torch.zeros((64, 220), dtype=torch.long)
    x = x.to(device=device)

    model = JigsawNet(*embed_mat.shape, 128, embed_mat)
    model = model.to(device=device)
    scores = model(x)
    print(scores.size())


def model_optimizer_init(nb_neurons, embed_mat, ft_lrs):
    model = JigsawNet(*embed_mat.shape, nb_neurons, embed_mat)

    params_emb = [p for p in model.emb_layer.parameters()]
    params_rnn = [p for p in model.rnns.parameters()]
    params_cls = [p for p in model.classifier.parameters()]

    optimizer = torch.optim.Adam(params=[{'params': params_emb, 'lr': ft_lrs[0]}])
    optimizer.add_param_group({'params':params_rnn, 'lr': ft_lrs[1]})
    optimizer.add_param_group({'params':params_cls, 'lr': ft_lrs[2]})

    return model, optimizer

