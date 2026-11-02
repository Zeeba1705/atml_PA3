import torch
import torch.nn as nn
import torch.nn.functional as F


class Connector(nn.Module):
    #g_phi: Linear 768 -> 1024, GELU, Linear 1024 -> 1024 (fp32, trainable in every stage)
    def __init__(self, vision_dim, hidden_dim, llm_dim):
        super().__init__()

        self.fc1= nn.Linear(vision_dim, hidden_dim)
        self.fc2= nn.Linear(hidden_dim, llm_dim)

        #fixed output scale from calibrate_scale (1.0 until calibrated); a buffer, so never trained
        self.register_buffer("scale", torch.ones(()))

    def forward(self, feats):
        #feats: (B, 196, 768) cached fp16 siglip features -> (B, 196, 1024) visual tokens
        return self.fc2(F.gelu(self.fc1(feats.to(self.fc1.weight.dtype)))) * self.scale


def mean_token_norm(x):
    #mean l2 norm over the last dim of all tokens
    return x.float().norm(dim=-1).mean()


def calibrate_scale(connector, train_feats, text_embeds):
    #rule: scale the initial connector output so its mean token norm equals the mean norm of text embeddings,
    #both measured on training data only. returns the scale and the two norms it came from
    with torch.no_grad():
        connector.scale.fill_(1.0)

        vis_norm= mean_token_norm(connector(train_feats))
        text_norm= mean_token_norm(text_embeds)

        scale= text_norm / vis_norm
        connector.scale.fill_(scale.item())

    return {"scale": scale.item(), "visual_norm_before": vis_norm.item(), "text_norm": text_norm.item()}
