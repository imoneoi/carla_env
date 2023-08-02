from typing_extensions import Concatenate
import torch
import torch.nn.functional as F
from torch import nn
import pdb

import compile_utils


class CompILE(nn.Module):
    """CompILE example implementation.

    Args:
        input_dim: Dictionary size of embeddings.
        hidden_dim: Number of hidden units.
        latent_dim: Dimensionality of latent variables (z).
        max_num_segments: Maximum number of segments to predict.
        temp_b: Gumbel softmax temperature for boundary variables (b).
        temp_z: Temperature for latents (z), only if latent_dist='concrete'.
        latent_dist: Whether to use Gaussian latents ('gaussian') or concrete /
            Gumbel softmax latents ('concrete').
    """
    def __init__(self, args, temp_b=1., temp_z=1.):
        super(CompILE, self).__init__()
        
        self.args = args

        self.state_dim = args.state_dim
        self.num_actions = args.num_actions # Discrete
        self.cont_action_dim = args.cont_action_dim # Continuous 
        self.hidden_dim = args.hidden_dim
        self.latent_dim = args.latent_dim
        self.max_num_segments = args.num_segments
        self.latent_dist = args.latent_dist
        self.temp_b = args.temp_b
        self.temp_z = args.temp_z

    
        if self.args.action_type == "continuous":
            if self.args.mode in ['statediff+action', 'state+action']:
                assert self.hidden_dim % 2 == 0
                self.encoder_embed_state = nn.Sequential(
                    nn.Linear(self.state_dim, self.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2)
                )
                self.encoder_embed_action = nn.Sequential(
                    nn.Linear(self.cont_action_dim, self.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2)
                )
                self.decoder_embed_state = nn.Sequential(
                    nn.Linear(self.state_dim, self.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2)
                )
                self.decoder_embed_action = nn.Sequential(
                    nn.Linear(self.cont_action_dim, self.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2)
                )
            elif self.args.mode == "action":
                self.encoder_embed_action = nn.Sequential(
                    nn.Linear(self.cont_action_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim)
                )
                self.decoder_embed_action = nn.Sequential(
                    nn.Linear(self.cont_action_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim)
                )
            else:
                raise ValueError('Invalid argument for `mode`.')
            
            self.decoder_head_action = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.cont_action_dim),
                nn.Tanh()
            )

        else:
            if self.args.mode in ['statediff+action', 'state+action']:
                assert self.hidden_dim % 2 == 0
                self.encoder_embed_state = nn.Sequential(
                    nn.Linear(self.state_dim, self.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2)
                )
                self.encoder_embed_action = nn.Embedding(self.num_actions + 1, self.hidden_dim // 2)  # +1 for padding

                self.decoder_embed_state = nn.Sequential(
                    nn.Linear(self.state_dim, self.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2)
                )
                self.decoder_embed_action = nn.Embedding(self.num_actions + 1, self.hidden_dim // 2)

            elif self.args.mode == "action":
                self.encoder_embed_action = nn.Embedding(self.num_actions + 1, self.hidden_dim)  # +1 for padding
                self.decoder_embed_action = nn.Embedding(self.num_actions + 1, self.hidden_dim)

            else:
                raise ValueError('Invalid argument for `mode`.')
            
            self.decoder_head_action = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.num_actions + 1)
            )

        if self.args.mode in ['statediff+action', 'state+action']:
            self.decoder_head_state = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.state_dim),
            )

        # Encoder LSTM
        self.encoder_lstm_cell = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

        # Encoder LSTM output heads.
        if self.latent_dist == "gaussian":
            self.head_z = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),  # Latents (z)
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.latent_dim * 2)
            )
        elif self.latent_dist == 'concrete':
            self.head_z = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),  # Latents (z)
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.latent_dim)
            )
        else:
            raise ValueError('Invalid argument for `latent_dist`.')

        self.head_b = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),  # Boundaries (b)
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

        # Decoder.
        self.decoder_1 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.decoder_1a = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.decoder_1b = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.decoder_lstm_cell = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

    def masked_encode(self, inputs, mask):
        """Run masked LSTM encoder on input sequence."""
        hidden = compile_utils.get_lstm_initial_state(
            inputs.size(0), self.hidden_dim, device=inputs.device)
        outputs = []
        for step in range(inputs.size(1)):
            hidden = self.encoder_lstm_cell(inputs[:, step], hidden)
            hidden = (mask[:, step, None] * hidden[0],
                      mask[:, step, None] * hidden[1])  # Apply mask.
            outputs.append(hidden[0])
        return torch.stack(outputs, dim=1)

    def get_boundaries(self, encodings, segment_id, lengths):
        """Get boundaries (b) for a single segment in batch."""

        if segment_id == self.max_num_segments - 1:
            # Last boundary is always placed on last sequence element.
            logits_b = None
            sample_b = torch.zeros_like(encodings[:, :, 0]).scatter_(
                1, lengths.unsqueeze(1) - 1, 1)
        else:
            logits_b = self.head_b(encodings).squeeze(-1)
            # Mask out first position with large neg. value.
            neg_inf = torch.ones(
                encodings.size(0), 1, device=encodings.device) * compile_utils.NEG_INF
            # Mask out padded positions with large neg. value.
            # TODO: Check off-by-one
            after_len_mask = (torch.arange(encodings.shape[1])[None, :].
                              to(self.args.device) > lengths[:, None]) * compile_utils.NEG_INF
            logits_b = torch.cat([neg_inf, logits_b[:, 1:]], dim=1)
            logits_b = logits_b + after_len_mask
            if self.training:
                sample_b = compile_utils.gumbel_softmax_sample(
                    logits_b, temp=self.temp_b)
            else:
                sample_b_idx = torch.argmax(logits_b, dim=1)
                sample_b = compile_utils.to_one_hot(sample_b_idx, logits_b.size(1))

        return logits_b, sample_b

    def get_latents(self, encodings, probs_b):
        """Read out latents (z) form input encodings for a single segment."""
        readout_mask = probs_b[:, 1:, None]  # Offset readout by 1 to left.
        readout = (encodings[:, :-1] * readout_mask).sum(1)
        logits_z = self.head_z(readout)

        # Gaussian latents.
        if self.latent_dist == 'gaussian':
            if self.training:
                mu, log_var = torch.split(logits_z, self.latent_dim, dim=1)
                sample_z = compile_utils.gaussian_sample(mu, log_var)
            else:
                sample_z = logits_z[:, :self.latent_dim]

        # Concrete / Gumbel softmax latents.
        elif self.latent_dist == 'concrete':
            if self.training:
                sample_z = compile_utils.gumbel_softmax_sample(
                    logits_z, temp=self.temp_z)
            else:
                sample_z_idx = torch.argmax(logits_z, dim=1)
                sample_z = compile_utils.to_one_hot(sample_z_idx, logits_z.size(1))
        else:
            raise ValueError('Invalid argument for `latent_dist`.')

        return logits_z, sample_z

    def decode(self, sample_z, length):
        """Decode single time step from latents and repeat over full seq."""
        hidden = F.relu(self.decoder_1(sample_z))
        pred = self.decoder_action_head(hidden)
        return pred.unsqueeze(1).repeat(1, length, 1)

    def decode_seq(self, sample_z, length):
        hidden = F.relu(self.decoder_1(sample_z))
        hidden = self.decoder_1a(hidden)
        cell = self.decoder_1b(hidden)
        hidden = (hidden, cell)
        
        batch_size = sample_z.shape[0]
        if self.args.action_type == "discrete":
            if self.args.mode in ['statediff+action', 'state+action']:
                start_action = torch.zeros((batch_size)).int().to(self.args.device)
                start_state = torch.zeros((batch_size, self.state_dim)).to(self.args.device)
            elif self.args.mode in ['action']:
                start_action = torch.zeros((batch_size)).int().to(self.args.device)
        else:
            if self.args.mode in ['statediff+action', 'state+action']:
                start_action = torch.zeros((batch_size, self.cont_action_dim)).to(self.args.device)
                start_state = torch.zeros((batch_size, self.state_dim)).to(self.args.device)
            elif self.args.mode in ['action']:
                start_action = torch.zeros((batch_size, self.cont_action_dim)).to(self.args.device)
        if self.args.mode in ['statediff+action', 'state+action']:
            start = torch.cat((self.decoder_embed_action(start_action), self.decoder_embed_state(start_state)), dim=-1)
        else:
            start = self.decoder_embed_action(start_action)
        
        outputs = []
        current = start
        for _ in range(length):
            hidden = self.decoder_lstm_cell(current, hidden)
            if self.args.mode in ['statediff+action', 'state+action']:
                out_action = self.decoder_head_action(hidden[0])
                out_state = self.decoder_head_state(hidden[0])
                out = [out_action, out_state]
            elif self.args.mode == 'action':
                out_action = self.decoder_head_action(hidden[0])
                out = out_action
            outputs.append(out)
            if self.args.action_type == "discrete":
                top_i_action = out_action.topk(1)[1].squeeze(-1)
                if self.args.mode in ['statediff+action', 'state+action']:
                    current_action_embed = self.decoder_embed_action(top_i_action.detach())
                    current_state_embed = self.decoder_embed_state(out_state.detach())
                    current = torch.cat((current_action_embed, current_state_embed), dim=-1)
                elif self.args.mode == 'action':
                    current_action_embed = self.decoder_embed_action(top_i_action.detach())
                    current = current_action_embed
            else:
                if self.args.mode in ['statediff+action', 'state+action']:
                    current_action_embed = self.decoder_embed_action(out_action.detach())
                    current_state_embed = self.decoder_embed_state(out_state.detach())
                    current = torch.cat((current_action_embed, current_state_embed), dim=-1)
                elif self.args.mode == 'action':
                    current_action_embed = self.decoder_embed_action(out_action.detach())
                    current = current_action_embed
        if self.args.mode in ['statediff+action', 'state+action']:
            # Predictions for both actions and states/statediffs
            return torch.stack([output[0] for output in outputs], dim=1), \
                    torch.stack([output[1] for output in outputs], dim=1)
        else:
             # Predictions for just actions (None for states/statesdiffs)
            return torch.stack(outputs, dim=1), None

    def get_next_masks(self, all_b_samples):
        """Get LSTM hidden state masks for next segment."""
        if len(all_b_samples) < self.max_num_segments:
            # Product over cumsums (via log->sum->exp).
            log_cumsums = list(
                map(lambda x: compile_utils.log_cumsum(x, dim=1), all_b_samples))
            mask = torch.exp(sum(log_cumsums))
            return mask
        else:
            return None

    def forward(self, states, actions, lengths):

        # Embed inputs.
        if self.args.mode == "action":
            action_embeddings = self.encoder_embed_action(actions)
            embeddings = action_embeddings
        elif self.args.mode == "statediff+action":
            action_embeddings = self.encoder_embed_action(actions)
            state_diff_embeddings = self.encoder_embed_state(states)
            embeddings = torch.cat((state_diff_embeddings, action_embeddings), dim=-1)
        elif self.args.mode == "state+action":
            action_embeddings = self.encoder_embed_action(actions)
            state_embeddings = self.encoder_embed_state(states)
            embeddings = torch.cat((state_embeddings, action_embeddings), dim=-1)

        # Create initial mask.
        mask = torch.ones(
            actions.size(0), actions.size(1), device=actions.device)

        all_b = {'logits': [], 'samples': []}
        all_z = {'logits': [], 'samples': []}
        all_encs = []
        all_recs = []
        all_masks = []
        for seg_id in range(self.max_num_segments):
            
            # Get masked LSTM encodings of inputs.
            encodings = self.masked_encode(embeddings, mask)
            all_encs.append(encodings)

            # Get boundaries (b) for current segment.
            logits_b, sample_b = self.get_boundaries(
                encodings, seg_id, lengths)
            all_b['logits'].append(logits_b)
            all_b['samples'].append(sample_b)

            # Get latents (z) for current segment.
            logits_z, sample_z = self.get_latents(
                encodings, sample_b)
            all_z['logits'].append(logits_z)
            all_z['samples'].append(sample_z)

            # Get masks for next segment.
            mask = self.get_next_masks(all_b['samples'])
            all_masks.append(mask)

            # Decode current segment from latents (z).
            reconstructions = self.decode_seq(sample_z, length=actions.size(1))
            all_recs.append(reconstructions)

        return all_encs, all_recs, all_masks, all_b, all_z
