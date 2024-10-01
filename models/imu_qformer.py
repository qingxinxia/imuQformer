import logging
import random
import einops
from conformer import Conformer
import torch
import torch.nn as nn
# from models.blip2 import Blip2Base
from models.Qformer import BertConfig, BertLMHeadModel
from models.imagebind_model import imagebind_model


class IMUQformer(nn.Module):

    @classmethod
    def init_imu_Qformer(cls, num_query_token,
                         vision_width,
                         num_hidden_layers=2):
        # bert model should be initialized in advance
        # but all the pretrained models are llm: https://huggingface.co/transformers/v3.3.1/pretrained_models.html
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        encoder_config.hidden_size = vision_width  # imu encoder output channel * (6-2)
        encoder_config.num_attention_heads = 8  # assert hidden_size % num_attention_heads == 0
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def __init__(self,
                 freeze_imunet=True,
                 freeze_qformer=False,
                 input_dim=6,
                 num_query_token=32,
                 ):
        super().__init__()

        self.num_imu_query_token = num_query_token
        # self.imu_encoder = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=1)
        # input shape is (batch, length, dim)
        self.imu_encoder = Conformer(input_dim=input_dim,
                                      encoder_dim=32,
                                      num_encoder_layers=3)
        self.imu_hidden_size = 32  # depending on length of imu_encoder
        # nn.Embedding has the same size of imu encoder
        self.imu_position_embedding = nn.Embedding(163, self.imu_hidden_size)

        self.imu_Qformer, self.imu_query_tokens = self.init_imu_Qformer(
            num_query_token=self.num_imu_query_token,
            vision_width=self.imu_hidden_size,
            num_hidden_layers=2)

        if freeze_qformer:
            for name, param in self.imu_position_embedding.named_parameters():
                param.requires_grad = False
            # logging.info('audio_Qformer and audio-LLAMA proj is frozen')
        else:
            for name, param in self.imu_position_embedding.named_parameters():
                param.requires_grad = True
            # logging.info('audio_Qformer is not frozen')

        if freeze_imunet:
            for name, param in self.imu_encoder.named_parameters():
                param.requires_grad = False
            # logging.info('audio_Qformer and audio-LLAMA proj is frozen')
        else:
            for name, param in self.imu_encoder.named_parameters():
                param.requires_grad = True

        # this layer connects the output of qformer and decoder (llm)
        self.imu_llm_proj = nn.Linear(5216, 1024)
        self.mse_loss = nn.MSELoss()

        return

    def encode_IMU_CNNqformer(self, imu):
        '''
        :param imu: shape (b,1,len,dim)
        :return:
        '''
        device = imu.device

        # simply use cnn to extract imu feature
        # imu_feature = self.imu_encoder(imu.unsqueeze(1))  # cnn layer
        input_lengths = torch.LongTensor([12345, 12300, 12000])  # random value
        imu_feature, _ = self.imu_encoder(imu, input_lengths)  # conformer: batch, 1/4length(45), dim32

        batch_size = imu_feature.shape[0]
        time_length = imu_feature.shape[1]
        imu_feature = imu_feature.reshape(batch_size, time_length, -1)

        position_ids = torch.arange(time_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        imu_position_embeddings = self.imu_position_embedding(position_ids)
        imu_bind_finalout = imu_feature + imu_position_embeddings

        # imu_query_tokens = self.audio_query_tokens.expand(imu_imagebind_finalout.shape[0], -1, -1)
        frame_atts = torch.ones(imu_bind_finalout.size()[:-1], dtype=torch.long).to(device)

        imu_query_output = self.imu_Qformer.bert(
            query_embeds=imu_bind_finalout,    # [batch, len163, dim32]
            encoder_hidden_states=imu_bind_finalout,
            encoder_attention_mask=frame_atts, # [batch, len163]
            return_dict=True,
        )
        imu_hidden = imu_query_output.last_hidden_state
        return imu_hidden  # [batch, len163, dim32]

    #  input audio shape [b t c h w]
    def encode_audioQformer(self, audio):
        device = audio.device
        with self.maybe_autocast():
            audio_feature, audio_imagebind_finalout = self.audio_encoder.get_audio_feature(audio, '')
            batch_size, time_length = audio.size()[:2]

            position_ids = torch.arange(time_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

            audio_position_embeddings = self.audio_position_embedding(position_ids)
            audio_imagebind_finalout = audio_imagebind_finalout + audio_position_embeddings

            audio_query_tokens = self.audio_query_tokens.expand(audio_imagebind_finalout.shape[0], -1, -1)
            frame_atts = torch.ones(audio_imagebind_finalout.size()[:-1], dtype=torch.long).to(device)

            audio_query_output = self.audio_Qformer.bert(
                query_embeds=audio_query_tokens,  # [32,768]
                encoder_hidden_states=audio_imagebind_finalout,
                encoder_attention_mask=frame_atts,
                return_dict=True,
            )
            audio_hidden = audio_query_output.last_hidden_state

            inputs_llama = self.audio_llama_proj(audio_hidden)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)

        return inputs_llama, atts_llama

    def forward(self, input, label_emb):

        imu_embeds = self.encode_IMU_CNNqformer(input)  # output: [batch, len163, dim32]

        # reshape imu enbedding (3D) to 2D
        imu_embeds_reshape = imu_embeds.reshape(imu_embeds.shape[0], -1)

        # prepare size to fit into llm
        out = self.imu_llm_proj(imu_embeds_reshape)

        # calculate contrastive loss between imu embedding and text embedding
        loss = self.mse_loss(out, label_emb)

        return loss
        # return {"loss": img_embeds}
