"""
PyTorch Lightning modules with training, validation,
testing and prediction loops and optimizers.
"""

import datetime
from timeit import default_timer as timer
import math

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, optim
from torch import LongTensor, BoolTensor, FloatTensor, Tensor

from syntheseus.reaction_prediction.models.smiles_medusa.tokenizer import InplaceSMILESTokenizer

class TokenEmbedding(nn.Module):
    """
    Embedding of token indices into a vector space.
    """

    def __init__(self, vocab_size: int, emb_size: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.emb_size = emb_size

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        return self.embedding(tokens)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, max_len: int = 5000):
        """
        Absolute positional encoding.
        Args
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return x


class VanillaTransformer(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            num_encoder_layers: int = 3,
            num_decoder_layers: int = 3,
            embedding_dim: int = 128,
            num_heads: int = 4,
            feedforward_dim: int = 256,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            src_pad_token_idx: int = 0,
            tgt_pad_token_idx: int = 0,
            custom_encoder: nn.Module | None = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.src_pad_token_i = src_pad_token_idx
        self.tgt_pad_token_i = tgt_pad_token_idx

        self.num_enc_layers = num_encoder_layers
        self.num_dec_layers = num_decoder_layers
        self.emb_dim = embedding_dim
        self.num_heads = num_heads
        self.ff_dim = feedforward_dim
        self.dropout_rate = dropout_rate
        self.activation = activation

        # Embedding constructor
        self.token_embedding = TokenEmbedding(
            self.vocab_size, self.emb_dim, padding_idx=self.src_pad_token_i
        )
        self.positional_encoding = PositionalEncoding(self.emb_dim)

        # Embedding updater
        layer_norm_eps = 1e-5
        batch_first = True
        norm_first = False
        # assert isinstance(custom_encoder, nn.Module) or custom_encoder is None
        if custom_encoder is None:
            _encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.emb_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.ff_dim,
                    dropout=self.dropout_rate,
                    activation=self.activation,
                    layer_norm_eps=layer_norm_eps,
                    batch_first=batch_first,
                    norm_first=norm_first,
                ),
                self.num_enc_layers,
                nn.LayerNorm(self.emb_dim, eps=layer_norm_eps),
            )
        else:
            _encoder = custom_encoder
        self.transformer = nn.Transformer(
            d_model=self.emb_dim,
            nhead=self.num_heads,
            batch_first=batch_first,
            custom_encoder=_encoder,
            custom_decoder=nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=self.emb_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.ff_dim,
                    dropout=self.dropout_rate,
                    activation=self.activation,
                    layer_norm_eps=layer_norm_eps,
                    batch_first=batch_first,
                    norm_first=norm_first,
                ),
                self.num_dec_layers,
                nn.LayerNorm(self.emb_dim, eps=layer_norm_eps),
            ),
        )

        # Decision function
        self.next_token_classifier = nn.Linear(
            self.emb_dim, self.vocab_size, bias=False
        )
        self.next_token_classifier.weight = self.token_embedding.embedding.weight

    def forward(self, src_token_ids: LongTensor, tgt_token_ids: LongTensor, src_mask=None):
        """
        Calculates the target decoder output embeddings
        Args:
            src_token_ids (LongTensor of size (b_sz, src_seq_len)): the token indices of the source sequences
            tgt_token_ids (LongTensor of size (b_sz, tgt_seq_len)): the token indices of the target sequences

        Returns:
            logits (FloatTensor of size (b_sz, tgt_seq_len, vocab_size)): the model output logits
        """

        tgt_emb = self.get_decoder_output_embs(src_token_ids, tgt_token_ids, src_mask=src_mask)

        logits = self.next_token_classifier(tgt_emb)
        return logits

    def encode_src(self, src_token_ids: LongTensor, src_pad_mask: BoolTensor, src_mask=None):
        # Embed tokens
        src_emb = self.positional_encoding(self.token_embedding(src_token_ids))

        # Update embeddings
        src_emb = self.transformer.encoder(src_emb, src_key_padding_mask=src_pad_mask, mask=src_mask)
        return src_emb

    def decode_tgt(
            self,
            tgt: LongTensor,
            memory: Tensor,
            memory_pad_mask: BoolTensor,
            src_mask=None
    ):

        tgt_emb = self.get_decoder_output_embs_using_src_memory(tgt, memory, memory_pad_mask, src_mask=src_mask)

        # Propose the next token
        logits = self.next_token_classifier(tgt_emb)
        return logits

    def get_decoder_output_embs(self, src_token_ids: LongTensor, tgt_token_ids: LongTensor, src_mask=None):
        """
        Calculates the target decoder output embeddings
        Args:
            src_token_ids (LongTensor of size (b_sz, src_seq_len)): the token indices of the source sequences
            tgt_token_ids (LongTensor of size (b_sz, tgt_seq_len)): the token indices of the target sequences

        Returns:
            tgt_emb (FloatTensor of size (b_sz, tgt_seq_len, emb_dim)): the decoder output
        """
        src_pad_mask: torch.Tensor = torch.where(
            src_token_ids != self.src_pad_token_i, torch.tensor(0.0), torch.tensor(float("-inf"))
        )
        memory = self.encode_src(src_token_ids, src_pad_mask, src_mask=src_mask)

        tgt_emb = self.get_decoder_output_embs_using_src_memory(tgt_token_ids, memory, src_pad_mask)
        return tgt_emb

    def get_decoder_output_embs_using_src_memory(self,
                                                 tgt_token_ids: LongTensor,
                                                 memory: Tensor,
                                                 memory_pad_mask: FloatTensor,
                                                 tgt_mask: FloatTensor | None = None,
                                                 tgt_src_mask: FloatTensor | None = None,
                                                 ) -> FloatTensor:
        """
        Calculates the target decoder output embeddings getting the source encoder output embeddings and
        the corresponding target indices
        Args:
            tgt_token_ids (LongTensor of size (b_sz, tgt_seq_len)): the token indices of the target sequences
            memory (FloatTensor of size (b_sz, src_seq_len, emb_dim)): the token indices of the target sequences
            memory_pad_mask (FloatTensor of size (b_sz, src_seq_len)): "-inf" indicates the place of pad tokens
            tgt_src_mask  (FloatTensor of size (b_sz, tgt_seq_len, src_seq_len))

        Returns:
            tgt_emb (FloatTensor of size (b_sz, tgt_seq_len, emb_dim)): the decoder output
        """

        # Embed tokens
        tgt_emb = self.positional_encoding(self.token_embedding(tgt_token_ids))

        # Update embeddings
        tgt_pad_mask = torch.where(tgt_token_ids != self.tgt_pad_token_i, torch.tensor(0.0), torch.tensor(float("-inf")))
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_token_ids.shape[1]).type_as(tgt_emb)
        if tgt_mask is not None:
            tgt_mask += tgt_mask

        tgt_emb = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
            memory_mask=tgt_src_mask
        )
        return tgt_emb


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class SmilesToSmilesAutoregressiveMedusaModel(pl.LightningModule):
    """
    The model translates source SMILES into target SMILES with an autoregressive encoder-decoder transformer.
    """

    def __init__(
            self,
            medusa_heads_number,
            medusa_layers_number,
            medusa_hidden_dim,
            embedding_dim: int = 128,  # Model arguments
            feedforward_dim: int = 256,
            num_encoder_layers: int = 3,
            num_decoder_layers: int = 3,
            transformer_heads_number: int = 4,
            dropout_rate: float = 0.0,
            activation: str = "relu",

            learning_rate: float = 3e-4,  # Optimization arguments
            weight_decay: float = 0.0,
            scheduler: str = "const",
            warmup_steps: int = 0,

            tokenizer: InplaceSMILESTokenizer | None = None,  # Prediction arguments
            beam_size: int = 1,
            max_size: int = 200,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.pad_token_idx = tokenizer.pad_token_idx
        self.bos_token_idx = tokenizer.bos_token_idx
        self.eos_token_idx = tokenizer.eos_token_idx
        self.C_token_idx = tokenizer.encoder_dict["C"]

        self.max_len = max_size
        self.beam_size = beam_size

        self.accepted_tokens_num = 0
        self.model_calls_num = 0

        self.log_prob_pad = 7.  # should be more than 0.

        self.produced_non_pad_tokens = 0
        self.classic_model_calls_num = 0

        self.acceptance_rate_pad_for_alredy_finished_seqs = -1  # should be negative
        self.acceptance_rate_pad_for_fake_seqs = -7  # should be negative
        self.model_config = {
            "vocab_size": self.vocab_size,
            "embedding_dim": embedding_dim,
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "num_heads": transformer_heads_number,
            "feedforward_dim": feedforward_dim,
            "dropout_rate": dropout_rate,
            "activation": activation,
        }

        self.optimization_config = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "scheduler": scheduler,
            "warmup_steps": warmup_steps,
        }

        self.save_hyperparameters(ignore=["tokenizer"])

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_idx,
                                             reduction='none')

        self.base_model = VanillaTransformer(**self.model_config)

        self.medusa_heads_num = medusa_heads_number  # M
        self.medusa_layers_number = medusa_layers_number
        self.medusa_hidden_dim = medusa_hidden_dim  # H
        self.act = nn.SiLU()
        self.norm_before_vocab = torch.nn.LayerNorm(
            embedding_dim, eps=1e-5, bias=True
        )
        # Create a list of Medusa heads
        self.medusa_heads = nn.Sequential(
            nn.Linear(embedding_dim, self.medusa_heads_num * self.medusa_hidden_dim),
            *([ResBlock(self.medusa_heads_num * self.medusa_hidden_dim)] * medusa_layers_number)
        )
        self.medusa_dim_restoration = nn.Linear(self.medusa_hidden_dim, embedding_dim)
        # Ensure medusa_head's dtype and device align with the base_model
        self.alpha = 0.  # length normalization parameter for beam search

    def forward(self, src_inds: torch.LongTensor, tgt_inds: torch.LongTensor, src_memory=None,
                memory_pad_mask=None) -> torch.Tensor:
        """
        B = batch size
        L = maximum sequence length defined in the data loader
        V = vocabulary size

        Args:
            src_inds (torch.LongTensor): Source sequence tensor of shape (B, Ls).
            tgt_inds (torch.LongTensor): Target sequence tensor of shape (B, L).

        Returns:
            torch.Tensor: Logits for the next token prediction of shape (B, L, M, V).
        """
        B, L = tgt_inds.size()
        M, H = self.medusa_heads_num, self.medusa_hidden_dim
        if src_memory is not None and memory_pad_mask is not None:
            pred_embs = self.base_model.get_decoder_output_embs_using_src_memory(tgt_inds, src_memory, memory_pad_mask)
        else:
            pred_embs = self.base_model.get_decoder_output_embs(src_inds, tgt_inds)  # -> (B, L, D)

        medusa_heads_embs_of_hidden_dim = self.medusa_heads(pred_embs).view(B, L, M, H)  # -> (B, L, M, H)
        medusa_heads_embs_of_hidden_dim = self.act(medusa_heads_embs_of_hidden_dim)
        medusa_heads_embs = self.medusa_dim_restoration(medusa_heads_embs_of_hidden_dim)  # -> (B, L, M, D)

        medusa_heads_embs = self.norm_before_vocab(medusa_heads_embs)
        logits = self.base_model.next_token_classifier(medusa_heads_embs)  # -> (B, L, M, V)
        return logits


    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        pass

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        pass

    def on_validation_batch_end(
            self,
            outputs: STEP_OUTPUT,
            batch: dict[str, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        pass

    def on_validation_epoch_end(self) -> None:
        pass

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        pass

    def predict_step(
            self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.LongTensor:
        """
        Predict the tokens of the target SMILES by unmasking the fully masked sequence.
        Source sequence goes into the encoder, fully masked target sequence goes into the decoder.
        Returns:
            torch.LongTensor: The predicted tokens of the target SMILES of shape (B, K, L).
            B = batch size, K = number of candidate sequences per source sequence, L = maximum sequence length.
        """
        src = batch["src_tokens"]
        generated = self.generate(src)
        return generated


    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.optimization_config["learning_rate"],
            weight_decay=self.optimization_config["weight_decay"],
            betas=(0.9, 0.999),
        )

        return optimizer

    def generate(self, src_token_ids: 'torch.LongTensor') -> tuple['torch.LongTensor', 'torch.Tensor']:
        b_size = src_token_ids.shape[0]

        n_drafts, draft_len = 1, self.medusa_heads_num

        src_pad_mask = (src_token_ids == self.pad_token_idx).bool()
        # -> (b_size, src_len)

        memory = self.base_model.encode_src(src_token_ids, src_pad_mask)
        # -> (b_size, src_len, emb_dim)
        _, src_len, emb_dim = memory.size()

        iters = -1

        generated_tokens = torch.full((b_size, 1), self.bos_token_idx, device=src_token_ids.device)
        #   -> (b_size, 1)
        
        draft_tokens_r = self(None, generated_tokens, src_memory=memory, memory_pad_mask=src_pad_mask).argmax(
            dim=-1)  # -> (b_size, 1, M)
        self.model_calls_num += 1
        log_probs = torch.full((b_size, 1), 0., device=src_token_ids.device)
        #   -> (b_size, 1)

        num_of_empty_columns = ((generated_tokens == self.pad_token_idx).sum(0) == b_size).sum().item()
        #   -> (1,)
        postn_after_the_last_meaning_token = generated_tokens.shape[1] - num_of_empty_columns
        #   -> (1,)
        possible_draft_len = self.max_len - postn_after_the_last_meaning_token - 1
        #   -> (b_size, 1)
        beam_size = 1

        logits_base = torch.full((b_size * n_drafts, draft_len + 1, self.vocab_size), 0., device=src_token_ids.device)
        #   -> (b_s * n_drafts, draft_len + 1, vocab_size)
        draft_tokens = torch.full((b_size, 1, draft_len), self.C_token_idx, device=src_token_ids.device)
        #   -> (b_s, 1, M)
        bool_idx_of_unfinished = ~((generated_tokens == self.eos_token_idx).sum(-1).bool())
        # -> (n_candidates)

        while possible_draft_len >= 1 and postn_after_the_last_meaning_token <= self.max_len:
            iters += 1
            logits_base = logits_base * 0.
            # We use artificial logits to avoid calculation of obvious pad predicting after eos
            logits_base[:, :, self.pad_token_idx] = 35.
            # 35. will give about 100% probability for pad_token after softmax()
            draft_len = min(possible_draft_len, draft_len)
            draft_tokens_r[:, :, :][draft_tokens_r[:, :, :] == self.eos_token_idx] = self.C_token_idx  # The drafts can't
            # be started with eos, otherwise it can lead to the repetitive answers
            draft_tokens_r[:, :, :][draft_tokens_r[:, :, :] == self.pad_token_idx] = self.C_token_idx
            draft_tokens[bool_idx_of_unfinished] = draft_tokens_r[:,:,:draft_tokens.shape[-1]]

            draft_tokens = draft_tokens[:, :, :draft_len]

            n_candidates, curr_len = generated_tokens.size()

            draft_place_len = draft_len + 1 - num_of_empty_columns
            if draft_place_len > 0:
                draft_place = torch.full((n_candidates, draft_place_len), self.pad_token_idx,
                                         device=src_token_ids.device)
                generated_tokens = torch.cat((generated_tokens, draft_place), dim=-1)
            # -> (n_candidates, drafted_len)
            logits_base = logits_base[:, :draft_len + 1, :]

            pad_place_bool = generated_tokens == self.pad_token_idx
            # -> (n_candidates, drafted_len)
            draft_place_bool = torch.logical_and(pad_place_bool,
                                                 pad_place_bool.cumsum(-1) <= draft_len)
            # -> (n_candidates, drafted_len)

            draft_place_bool_idx_input = draft_place_bool.unsqueeze(1).repeat(1, n_drafts, 1)
            # -> (b_s * bm_sz, n_drafts, drafted_len)
            generated_tokens_input = generated_tokens.unsqueeze(1).repeat(1, n_drafts, 1)
            # -> (b_s * bm_sz, n_drafts, drafted_len)

            generated_tokens_input[draft_place_bool_idx_input] = draft_tokens.reshape(-1)
            draft_place_bool_idx_input = draft_place_bool_idx_input.flatten(end_dim=1)
            # -> (b_s * bm_sz * n_drafts, drafted_len)
            generated_tokens_input = generated_tokens_input.flatten(end_dim=1)
            # # -> (b_s * bm_sz * n_drafts, drafted_len)

            bool_idx_of_unfinished = bool_idx_of_unfinished.unsqueeze(-1).repeat(1, n_drafts).flatten(end_dim=1)
            # -> (b_s * bm_sz * n_drafts)
            draft_place_bool_idx_input = draft_place_bool_idx_input[bool_idx_of_unfinished]
            #   -> (num_of_unfinished, drafted_len)
            
            pred_logits = self(None,
                               generated_tokens_input[bool_idx_of_unfinished],
                               src_memory=memory[bool_idx_of_unfinished],
                               memory_pad_mask=src_pad_mask[bool_idx_of_unfinished])[:, :, 0, :]
            #  -> (num_of_unfinished, drafted_len, vocab_size)
            self.model_calls_num += 1
            vocab_size = pred_logits.shape[-1]

            pred_logits = pred_logits[
                torch.logical_or(draft_place_bool_idx_input, torch.roll(draft_place_bool_idx_input, -1, 1))].reshape(
                -1, draft_len + 1, vocab_size)
            #  -> (num_of_unfinished, draft_len + 1, vocab_size)

            logits_base[bool_idx_of_unfinished] = pred_logits
            pred_logits = logits_base
            #   -> (n_candidates * n_drafts, draft_len + 1, vocab_size)

            # Choosing the best draft for each candidate. The draft with the biggest number of
            # approved tokens is the best draft for the given candidate. #########################################

            # All unapproved tokens in masked_probs have zero probability
            # We use nucleus=0.9975 and max_num_of_unmasked_positions=beam_size to avoid sampling of low probable sequences
            # and reduce calculation
            masked_probs = mask_with_num_logits_according_nucleus(pred_logits, nucleus=0.9975,
                                                                  max_num_of_unmasked_positions=self.beam_size,
                                                                  num="-inf").softmax(-1)
            #   -> (n_candidates * n_drafts, draft_len + 1, vocab_size)

            masked_probs = masked_probs.reshape(n_candidates, n_drafts, draft_len + 1, vocab_size)
            draft_tokens = draft_tokens.reshape(n_candidates, n_drafts, draft_len)

            n_accepted_in_drafts = self.calculate_n_accepted_in_drafts(draft_tokens, masked_probs)
            #   ->(n_candidates, n_drafts)

            # Each candidate needs its best draft. Choose the draft with the biggest number of approved tokens
            # for each candidate:
            n_accepted, draft_i = n_accepted_in_drafts.topk(1, dim=-1)
            # (n_candidates, n_drafts) -> (n_candidates, 1)

            chosen_drafts = torch.gather(draft_tokens, dim=1,
                                         index=draft_i.unsqueeze(-1).expand(n_candidates, 1, draft_len)).squeeze(1)
            #   -> (n_candidates, draft_len)
            ########################################################################################################
            pred_logits = pred_logits.reshape(n_candidates, n_drafts, draft_len + 1, vocab_size)

            # Further we need information only about chosen drafts
            pred_logits = torch.gather(pred_logits, dim=1, index=draft_i.unsqueeze(-1).unsqueeze(-1).
                                       expand(n_candidates, 1, draft_len + 1, vocab_size)).squeeze(1)
            #   -> (n_candidates, draft_len + 1, vocab_size)

            # Sample all possible lines within the chosen drafts:
            # new_candidates have the initial tokens and the new ones

            new_candidates, new_log_probs, num_of_new_seqs_for_each_in_batch, accepted_tokens_num,  det_new_seqs_log_probs = \
                self.sample(generated_tokens, log_probs, pred_logits,
                            chosen_drafts, b_size, draft_place_bool, n_accepted.squeeze(-1))
            new_log_probs_normalized = new_log_probs / (new_candidates != self.pad_token_idx).sum(-1).unsqueeze(-1).pow(self.alpha)

            _, top_inds_1d = topk_in_each_group(score_1d=new_log_probs_normalized,
                                                            length_of_each_group=num_of_new_seqs_for_each_in_batch,
                                                            k=self.beam_size, pad=-float("inf"))
            new_log_probs = new_log_probs[top_inds_1d].reshape(num_of_new_seqs_for_each_in_batch.shape[0], self.beam_size)
            # -> (b_size, beam_size)

            new_candidates = new_candidates[top_inds_1d]
            # -> (b_size * beam_size, drafted_len)

            accepted_tokens_num = accepted_tokens_num[top_inds_1d]
            # -> (b_size * beam_size,)
            accepted_tokens_num = accepted_tokens_num[accepted_tokens_num >= 0]

            self.accepted_tokens_num += accepted_tokens_num.sum().item()

            self.produced_non_pad_tokens += accepted_tokens_num.sum().item() + accepted_tokens_num.shape[0]

            if (new_candidates == self.eos_token_idx).sum(-1).bool().sum() == b_size * self.beam_size:
                break
            generated_tokens = new_candidates
            bool_idx_of_unfinished = ~((generated_tokens == self.eos_token_idx).sum(-1).bool())
            # -> (n_candidates)
            inds_for_next_drafts = (generated_tokens != self.pad_token_idx).sum(-1) - 1  # -> (b_s * bm_sz)
            if iters == 0:
                src_pad_mask = src_pad_mask.unsqueeze(1).repeat(1, self.beam_size, 1).flatten(end_dim=1)
                # -> (b_size * n_drafts * bm_sz, src_len)
                memory = memory.unsqueeze(1).repeat(1, self.beam_size, 1, 1).flatten(end_dim=1)
                # -> (b_size * n_drafts * bm_sz, src_len, emb_dim)

                logits_base = logits_base.repeat(self.beam_size, 1, 1)
                #   -> (b_s * n_drafts * bm_sz, draft_len + 1, vocab_size)

                draft_tokens = draft_tokens.repeat(self.beam_size, 1, 1)
                #   -> (b_s * bm_sz, 1, M)
            draft_tokens_r = self(None,
                                generated_tokens[bool_idx_of_unfinished],
                                src_memory=memory[bool_idx_of_unfinished],
                                memory_pad_mask=src_pad_mask[bool_idx_of_unfinished]).argmax(-1)
            # -> (b_s * bm_sz, drafted_len, M)
            self.model_calls_num += 1
            draft_tokens_r = torch.gather(draft_tokens_r, dim=1,
                                        index=inds_for_next_drafts[bool_idx_of_unfinished].unsqueeze(-1).unsqueeze(-1).repeat(1, 1,
                                                                                                      self.medusa_heads_num))
            # -> (b_s * bm_sz, 1, M)

            log_probs = new_log_probs.reshape(b_size * self.beam_size, 1)
            # -> (b_size * beam_size, 1)

            num_of_empty_columns = torch.min((generated_tokens == self.pad_token_idx).sum(-1)).item()
            #   -> (1,)
            postn_after_the_last_meaning_token = generated_tokens.shape[1] - num_of_empty_columns
            #   -> (1,)
            possible_draft_len = self.max_len - postn_after_the_last_meaning_token - 1
            #   -> (b_size, 1)
        self.classic_model_calls_num += (new_candidates != self.pad_token_idx).sum(-1).max().item() - 1
        return new_candidates.reshape(b_size, self.beam_size, -1), new_log_probs.reshape(b_size, self.beam_size).exp()

    def calculate_n_accepted_in_drafts(self, draft_tokens, masked_probs):
        """
        This function calculates the number of approved tokens in each draft for each candidate.

        :param draft_tokens: tensor of size (n_candidates, n_drafts, draft_len),
        :param masked_probs: (all unapproved tokens in masked_probs are supposed to be equal to 0.)
                             tensor of size (n_candidates, n_drafts, draft_len + 1, vocab_size),

        :return:
          ->  returns the number of approved tokens in each draft for each candidate:
                             tensor of size  (n_candidates, n_drafts)

        """
        draft_tokens_probs = torch.gather(masked_probs[:, :, :-1, :], dim=-1,
                                          index=draft_tokens.unsqueeze(-1)).squeeze(
            -1)
        #   -> (n_candidates, n_drafts, draft_len)
        verification = draft_tokens_probs != 0.

        _range = verification.cumsum(-1)  # (n_candidates, n_drafts, draft_len)
        accepted_in_drafts_bool = (torch.arange(1, verification.size(2) + 1).unsqueeze(0).unsqueeze(0).type_as(
            _range) == _range)  # (n_candidates, n_drafts, draft_len)

        return accepted_in_drafts_bool.sum(-1)  # (n_candidates, n_drafts, draft_len) -> (n_candidates, n_drafts)

    def sample(self, curr_lines, curr_log_probs, pred_logits, chosen_drafts, b_size, draft_place_bool, n_accepted):
        """
        This function samples all possible sequences within a selected draft. Each draft can
        produce (self.max_num_positions_for_sampling - 1) * num_of_approved_tokens + self.max_num_positions_for_sampling
        at most.

        :param curr_lines: tensor (n_candidates, drafted_len),
        :param curr_log_probs_history: tensor (n_candidates, 1),
        :param pred_logits: tensor (n_candidates, draft_len + 1, vocab_size),
        :param chosen_drafts: tensor (n_candidates, draft_len),
        :param b_size: int,
        :param draft_place_bool: tensor (n_candidates, drafted_len), it contains true where the draft supposed to be in curr_lines,
            in each line there are draft_len trues
        :param n_accepted: tensor (n_candidates)
        :return:
          ->  new_lines: tensor (num_lines, len),
              new_log_probs: tensor (num_lines, 1)
              num_of_new_seqs_for_each_in_batch: tensor (b_size)
              token_postn: tensor (num_lines), to calculate the number of accepted tokens in the next top n sequences
                later; self.acceptance_rate_pad_for_already_finished_seqs means that the given sequence had already the
                eos token and so didn't need subsequent tokens
        """
        n_candidates, draft_len_plus_one, vocab_size = pred_logits.size()

        draft_len = draft_len_plus_one - 1

        masked_logits = mask_with_num_logits_according_nucleus(pred_logits, nucleus=20.,
                                                               max_num_of_unmasked_positions=self.beam_size,
                                                               num=0.)
        #   -> (n_candidates, draft_len + 1, vocab_size)
        # any nucleus more than 1. fits well
        tmp_range = torch.arange(draft_len_plus_one).type_as(curr_lines).unsqueeze(0)
        #   -> (1, draft_len + 1)
        ####################################################################################

        mask_for_unaccepted_draft_tokens = tmp_range.repeat(n_candidates, 1) <= n_accepted.unsqueeze(-1)
        #   -> (n_candidates, draft_len + 1)
        masked_logits *= mask_for_unaccepted_draft_tokens.unsqueeze(-1)

        not_entirely_excepted_bool = n_accepted != draft_len
        #   -> (n_candidates)
        if not_entirely_excepted_bool.sum().item() > 0:
            # We need to change the first token in the drafts, following the last accepted token, to the bos token in
            # order to build the right top n tree of sequences
            chosen_drafts[not_entirely_excepted_bool] = chosen_drafts[not_entirely_excepted_bool].scatter(
                index=n_accepted[not_entirely_excepted_bool].unsqueeze(-1), dim=1, value=self.bos_token_idx)

        masked_logits[:, :-1, :].scatter_(index=chosen_drafts.unsqueeze(-1), dim=2, value=0.)  # the accepted tokens in
        # the drafts can not be leaves of the top n tree of the sequences

        # Sampling the top n tree of sequences leaves:
        candts_inds, token_postn, token_inds = torch.nonzero(masked_logits, as_tuple=True)
        # -> (num)

        ################################################################################################################
        if n_candidates == b_size:
            beam_size = 1
        else:
            beam_size = self.beam_size
        assert n_candidates / b_size == beam_size
        candts_inds_tmp = candts_inds.unsqueeze(-1).repeat(1, b_size)
        #  -> (b_size * beam_size, b_size)
        low_border = torch.arange(b_size).to(candts_inds.device) * beam_size
        high_border = low_border + beam_size
        num_of_new_seqs_for_each_in_batch = torch.logical_and(low_border <= candts_inds_tmp,
                                                              candts_inds_tmp < high_border).sum(0)
        # -> (b_size)
        ################################################################################################################

        num = token_inds.size()[0]
        previous_roots = curr_lines[candts_inds]  # (num, drafted_len)
        already_finished_given_seqs = (previous_roots == self.eos_token_idx).sum(-1).bool()  # -> (num)

        log_prob_of_roots = curr_log_probs[candts_inds]  # (num, 1)
        draft_place_bool = draft_place_bool[candts_inds]  # (num, max_len)

        drafts = chosen_drafts[candts_inds]  # (num, draft_len)
        tail = torch.full((num, 1), 0.).type_as(drafts)  # -> (num, 1)
        new_seqs = torch.cat((drafts, tail), dim=-1)  # (num, draft_len+1)
        new_seqs.scatter_(1, index=token_postn.unsqueeze(-1), src=token_inds.unsqueeze(-1))
        #   -> (num, draft_len + 1)

        mask_for_tokens_after_the_sampled = tmp_range > token_postn.unsqueeze(-1)
        #   -> (num, draft_len + 1)
        predicted_log_probs = pred_logits.softmax(-1).log()[candts_inds]  # -> (num, draft_len + 1, vocab_size)

        new_seqs_log_probs = torch.gather(predicted_log_probs, dim=2, index=new_seqs.unsqueeze(-1)).squeeze(-1)
        #    -> (num, draft_len + 1)
        new_seqs_log_probs.masked_fill_(mask_for_tokens_after_the_sampled, 0.)
        #    -> (num, draft_len + 1)
        new_seqs_log_probs_C = new_seqs_log_probs.cumsum(dim=-1)  # -> (num, draft_len + 1)

        last_log_prob_from_roots = torch.min(log_prob_of_roots, dim=-1, keepdim=True).values
        # (num, 1)
        new_seqs_log_probs_1 = last_log_prob_from_roots + new_seqs_log_probs_C[:, -1:]
        #    -> (num, 1)
        new_seqs.masked_fill_(mask_for_tokens_after_the_sampled, self.pad_token_idx)
        #    -> (num, draft_len + 1)

        new_seqs_place_bool = torch.logical_or(draft_place_bool, torch.roll(draft_place_bool, 1, 1))
        # -> (num, drafted_len) It contains draft_len+1 Trues in each line
        previous_roots[new_seqs_place_bool] = new_seqs.reshape(-1)

        token_postn[already_finished_given_seqs] = self.acceptance_rate_pad_for_alredy_finished_seqs
        # the given sequences with eos didn't need the draft tokens. We
        # don't take pads into account calculating the acceptance rate
        return previous_roots, new_seqs_log_probs_1, num_of_new_seqs_for_each_in_batch, token_postn, new_seqs_log_probs


def topk_in_each_group(score_1d, length_of_each_group, k, pad=None):
    """
    This function finds the biggest k values and the corresponding indices. It is needed when each group has different
    number of candidates.
    N - number of_groups.

    :param score_1d: tensor (shape_0 = sum(length_of_each_group), 1)
    :param length_of_each_group: tensor (N,), Each length should be >= k
    :param k: int
    :param pad: it's needed to fill fake score_1d positions to make reshape (N, max_len_of_group) possible. Pad should
    be less than each number in score_1d

    :return:
      ->  topk_score: tensor (N, k),
          topk_inds_1d: tensor (N * k,); score_1d[topk_inds_1d].reshape(N, k) is equal to topk_score.

    """
    b_size = length_of_each_group.shape[0]
    assert torch.min(length_of_each_group).item() >= k
    max_len_of_group = torch.max(length_of_each_group).item()

    # We make fake sequences with an artificial probability -inf in case if a different number of sequences
    # were sampled on the basis of the chosen drafts
    start_ind_of_each_group = torch.roll(length_of_each_group, 1, dims=-1)  # -> (b_size)
    start_ind_of_each_group[0] = 0
    start_ind_of_each_group = start_ind_of_each_group.cumsum(-1).unsqueeze(1)
    # -> (N, 1)

    different_num_of_candidates_in_groups = (length_of_each_group == max_len_of_group).sum() != b_size
    if different_num_of_candidates_in_groups:
        if pad is None:
            pad = torch.min(score_1d).item() - 1

        inds_for_2d = torch.arange(max_len_of_group).to(score_1d.device).unsqueeze(0).repeat(b_size, 1)
        # -> (N, max_len_of_group)

        mask_for_fake_seqs = inds_for_2d >= length_of_each_group.unsqueeze(1)
        inds_for_2d = start_ind_of_each_group + (inds_for_2d % length_of_each_group.unsqueeze(1))

        score_1d = score_1d[inds_for_2d.reshape(-1)]
        # -> (N * max_len_of_group, 1)

        score_1d[mask_for_fake_seqs.reshape(-1)] = pad  # pads
        score_2d = score_1d.reshape(b_size, max_len_of_group)
        # -> (N, max_len_of_group)

        topk_score, topk_inds = score_2d.topk(k=k, axis=-1, sorted=True)
        # -> (N, k)

        topk_inds_1d = torch.gather(inds_for_2d, dim=1, index=topk_inds)
        #  (N, max_len_of_group) -> (N, k)
    else:
        score_2d = score_1d.reshape(b_size, max_len_of_group)
        # -> (N, max_len_of_group)

        topk_score, topk_inds = score_2d.topk(k=k, axis=-1, sorted=True)
        # -> (N, k)

        topk_inds_1d = start_ind_of_each_group + topk_inds
    topk_inds_1d = topk_inds_1d.reshape(-1)
    #  -> (N * k,)
    return topk_score, topk_inds_1d


def mask_with_num_logits_according_nucleus(pred_logits, nucleus, max_num_of_unmasked_positions, num=0.):
    """
    This function fills all unapproved tokens' logits with float(num). It uses nucleus parameter to decide which logits
    are big enough. No more than max_num_of_unmasked_positions but at least the best logit will be left unmasked
    for each distribution.
    If nucleus < 0, then it works in greedy mode. It masks everything accept the best token in each distribution.
    If nucleus > 1, then it works in beam search mode. It masks nothing and chooses the top n tokens in each distribution,
        where n is equal to max_num_of_unmasked_positions.
    If 0 < nucleus < 1 (we recommend nucleus = 0.9975), it works in top k mode. It masks all tokens' logits with
    cumulative probability above or equal to the nucleus parameter. But no more than max_num_of_unmasked_positions will
    be left unmasked in each row.
    """
    n_candidates, curr_len, vocab_size = pred_logits.size()  # (n_candidates, draft_len + 1, vocab_size)
    pred_logits = pred_logits.reshape(n_candidates * curr_len,
                                      vocab_size)  # -> (n_candidates * curr_len, vocab_size)

    sorted_logits, sorted_indices = torch.sort(pred_logits,
                                               descending=True)  # -> (n_candidates * curr_len, vocab_size)
    cumulative_probs = torch.cumsum(sorted_logits.softmax(-1), dim=-1)  # -> (n_candidates * curr_len, vocab_size)

    cumulative_probs = torch.roll(cumulative_probs, 1, dims=-1)

    cumulative_probs[:, 0] = nucleus - 1  # this protects the best probability in each distribution
    # Remove tokens with cumulative probability above or equal to the threshold (nucleus parameter).
    # At least the best probability in each row will be left unmasked
    keep_candidates_mask = cumulative_probs < nucleus  # -> (n_candidates * curr_len, vocab_size)

    keep_candidates_mask[:, max_num_of_unmasked_positions:] = False  # no more than max_num_of_unmasked_positions

    sorted_logits.masked_fill_(~keep_candidates_mask, float(num))  # the all unapproved tokens logits
    # will be set equal to float(num)

    masked_logits_according_nucleus = torch.gather(sorted_logits, 1, sorted_indices.argsort(1))
    # -> (n_candidates * curr_len, vocab_size)
    return masked_logits_according_nucleus.reshape(n_candidates, curr_len, vocab_size)
