import torch
from torch import nn, Tensor
import logging
from conformer import Conformer

LARGE_NUM = 1e9


class MLP1(nn.Module):
    def __init__(self, hidden_dim=2048, norm=None, activation="relu"):  # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        # if activation == "relu":
        #     activation_layer = nn.ReLU()
        # elif activation == "leakyrelu":
        #     activation_layer = nn.LeakyReLU()
        # elif activation == "tanh":
        #     activation_layer = nn.Tanh()
        # elif activation == "sigmoid":
        #     activation_layer = nn.Sigmoid()
        # else:
        #     raise ValueError(f"Unknown activation function")

        if norm:
            if norm == 'bn':
                norm_layer = nn.BatchNorm1d
            else:
                norm_layer = nn.LayerNorm

            self.layer1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                norm_layer(hidden_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.layer1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True)
            )

        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SimCLRLoss(nn.Module):
    """
    https://github.com/Nicolik/SimpleCNNClassifier/blob/master/train.py
    This loss was used in our SBERT publication (https://arxiv.org/abs/1908.10084) to train the SentenceTransformer
    model on NLI data. It adds a softmax classifier on top of the output of two transformer networks.

    :param model: SentenceTransformer model
    :param sentence_embedding_dimension: Dimension of your sentence embeddings
    :param num_labels: Number of different labels
    :param concatenation_sent_rep: Concatenate vectors u,v for the softmax classifier?
    :param concatenation_sent_difference: Add abs(u-v) for the softmax classifier?
    :param concatenation_sent_multiplication: Add u*v for the softmax classifier?

    Example::

        from sentence_transformers import SentenceTransformer, SentencesDataset, losses
        from sentence_transformers.readers import InputExample

        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        train_examples = [InputExample(InputExample(texts=['First pair, sent A', 'First pair, sent B'], label=0),
            InputExample(texts=['Second Pair, sent A', 'Second Pair, sent B'], label=3)]
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)
    """

    def __init__(self,
                 input_dim: int = 6,
                 embedding_dimension: int = 32,
                 # num_labels: int,
                 concatenation_sent_rep: bool = True,
                 concatenation_sent_difference: bool = True,
                 concatenation_sent_multiplication: bool = False,
                 concatenation_sent_max_square: bool = False,  # 拼接两个句子表示的max-square（如寐建议的一个trick）
                 projection_norm_type: str = "ln",
                 do_hidden_normalization: bool = True,  # 进行对比损失之前，是否对句子表示做正则化
                 temperature: float = 1.0,  # 对比损失中的温度系数，仅对于交叉熵损失有效
                 add_contrastive_predictor: bool = True,
                 ):
        super(SimCLRLoss, self).__init__()
        self.model = Conformer(input_dim=input_dim,
                              encoder_dim=embedding_dimension,
                              num_encoder_layers=3)
        # self.num_labels = num_labels
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication
        self.concatenation_sent_max_square = concatenation_sent_max_square

        self.do_hidden_normalization = do_hidden_normalization
        self.temperature = temperature
        self.add_contrastive_predictor = add_contrastive_predictor
        if add_contrastive_predictor:
            self.predictor = MLP1(hidden_dim=embedding_dimension, norm=projection_norm_type)


    def _reps_to_output(self, rep_a: torch.Tensor, rep_b: torch.Tensor):
        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        if self.concatenation_sent_max_square:
            vectors_concat.append(torch.max(rep_a, rep_b).pow(2))

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)
        return output

    def _contrastive_loss_forward(self,
                                  hidden1: torch.Tensor,
                                  hidden2: torch.Tensor,
                                  hidden_norm: bool = True,
                                  temperature: float = 1.0):
        """
        hidden1/hidden2: (bsz, dim)
        """
        batch_size, hidden_dim = hidden1.shape[0], hidden1.shape[1]*hidden1.shape[-1]
        hidden1 = hidden1.reshape(batch_size, -1)
        hidden2 = hidden2.reshape(batch_size, -1)

        if hidden_norm:
            hidden1 = torch.nn.functional.normalize(hidden1, p=2, dim=-1)
            hidden2 = torch.nn.functional.normalize(hidden2, p=2, dim=-1)

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = torch.arange(0, batch_size).to(device=hidden1.device)
        masks = torch.nn.functional.one_hot(torch.arange(0, batch_size), num_classes=batch_size).to(
            device=hidden1.device, dtype=torch.float)

        logits_aa = torch.matmul(hidden1, hidden1_large.transpose(0, 1)) / temperature  # shape (bsz, bsz)
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.transpose(0, 1)) / temperature  # shape (bsz, bsz)
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.transpose(0, 1)) / temperature  # shape (bsz, bsz)
        logits_ba = torch.matmul(hidden2, hidden1_large.transpose(0, 1)) / temperature  # shape (bsz, bsz)

        loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
        loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
        loss = loss_a + loss_b
        return loss

    def forward(self, view1, view2):
        input_lengths = torch.LongTensor([12345, 12300, 12000])  # random value
        rep_a_view1, _ = self.model(view1, input_lengths)
        rep_a_view2, _ = self.model(view2, input_lengths)

        # add predictor
        if self.add_contrastive_predictor:
            rep_a_view1 = self.predictor(rep_a_view1)
            rep_a_view2 = self.predictor(rep_a_view2)

        # final_loss = 0

        contrastive_loss = self._contrastive_loss_forward(rep_a_view1, rep_a_view2,
                                                          hidden_norm=self.do_hidden_normalization,
                                                          temperature=self.temperature)
        # self.model.tensorboard_writer.add_scalar(f"train_contrastive_loss", contrastive_loss.item(),
        #                                          global_step=self.model.global_step)
        # final_loss += contrastive_loss
        # self.model.tensorboard_writer.add_scalar(f"train_contrastive_loss_total", contrastive_loss.item(),
        #                                          global_step=self.model.global_step)

        return contrastive_loss