""" Define the network architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as np


class EncoderRNN(nn.Module):
    """
    A bidirectional RNN. It takes FBANK features and outputs the output state vectors of every time step.
    """
    def __init__(self, hidden_size, num_layers, use_bn):
        """
        Args:
            hidden_size (integer): Size of GRU cells.
            num_layers (integer): Number of GRU layers.
            use_bn (bool): Whether to insert BatchNorm in each layer.
        """
        super(EncoderRNN, self).__init__()
        self.embed = nn.Linear(240, hidden_size)   # 240 is the dimension of acoustic features.
        self.rnns = nn.ModuleList([])
        if use_bn:
            self.bns = nn.ModuleList([])
        for i in range(num_layers):
            if i == 0:
                insize = hidden_size
            else:
                insize = 2 * hidden_size
            self.rnns.append(nn.GRU(insize, hidden_size, batch_first=True, bidirectional=True))
            if hasattr(self, 'bns'):
                self.bns.append(nn.BatchNorm1d(insize))
        # The initial state is a trainable vector.
        self.init_state = torch.nn.Parameter(torch.randn([2 * num_layers, 1, hidden_size]))
        self.num_layers = num_layers

    def forward(self, xs, xlens):
        """
        We pack the padded sequences because it is especially important for bidirectional RNN to work properly. The RNN 
        in opposite direction can ignore the first few <PAD> tokens after packing.

        Args:
            xs (torch.FloatTensor, [batch_size, seq_length, dim_features]): A mini-batch of FBANK features.
            xlens (torch.LongTensor, [batch_size]): Sequence lengths before padding.

        Returns:
            xs (PackedSequence): The packed output states.
        """
        batch_size = xs.shape[0]
        xs = self.embed(xs)
        xs = rnn_utils.pack_padded_sequence(xs,
                                            xlens,
                                            batch_first=True,
                                            enforce_sorted=False)
        inits = self.init_state.repeat([1, batch_size, 1])   # [2 * num_layers, batch_size, hidden_size]
        for i in range(self.num_layers):
            if hasattr(self, 'bns'):
                xs = self.apply_bn(xs, i)
            xs, _ = self.rnns[i](xs, inits[i*2:(i+1)*2])
        return xs

    def apply_bn(self, xs, layer_id):
        """
        BatchNorm forward pass.

        Args:
            xs (PackedSequence): Packed input sequence.
            layer_id (integer): Which layer it is working on.

        Returns:
            xs (PackedSequence): Packed sequence after applying BatchNorm.
        """
        # Unpack
        xs, xlens = rnn_utils.pad_packed_sequence(
            xs, batch_first=True)   # [batch_size, padded_seq_length, C], [batch_size]
        # Ignore zero paddings
        batch_size = xs.shape[0]
        xs = [xs[i, :xlens[i]] for i in range(batch_size)]
        # Concatenate
        xs = torch.cat(xs, dim=0)     # [total_seq_length, C]
        # Apply BatchNorm
        xs = self.bns[layer_id](xs)   # [total_seq_length, C]
        # Repack
        xlens = [0] + torch.cumsum(xlens, dim=0).tolist()   # [batch_size + 1]
        xs = [xs[xlens[i]:xlens[i+1]] for i in range(batch_size)]
        xs = rnn_utils.pack_sequence(xs, enforce_sorted=False)
        return xs


class MultiLayerGRUCell(nn.Module):
    """
    Stack multiple GRU cells. For DecoderRNN.
    """
    def __init__(self, input_size, hidden_size, num_layers, drop_p):
        """
        Args:
            input_size (integer): Input size of GRU cells.
            hidden_size (integer): Hidden layer size of GRU cells.
            num_layers (integer): Number of GRU layers.
            drop_p (float): Probability to drop elements at Dropout layers.
        """
        super(MultiLayerGRUCell, self).__init__()

        self.cells = nn.ModuleList([])
        for i in range(num_layers):
            if i==0:
                self.cells.append(nn.GRUCell(input_size, hidden_size))
            else:
                self.cells.append(nn.GRUCell(hidden_size, hidden_size))
        self.dropouts = nn.ModuleList([nn.Dropout(drop_p) for _ in range(num_layers-1)])
        self.num_layers = num_layers

    def forward(self, x, h):
        """
        One step forward pass.
        
        Args:
            x (torch.FloatTensor, [batch_size, input_size]): The input features of current time step.
            h (torch.FloatTensor, [num_layers, batch_size, hidden_size]): The hidden state of previous time step.
            
        Returns:
            outputs (torch.FloatTensor, [num_layers, batch_size, hidden_size]): The hidden state of current time step.
        """
        outputs = []
        for i in range(self.num_layers):
            if i==0:
                x = self.cells[i](x, h[i])
            else:
                x = self.cells[i](self.dropouts[i-1](x), h[i])
            outputs.append(x)
        outputs = torch.stack(outputs, dim=0)
        return outputs


class DecoderRNN(nn.Module):
    """
    A decoder network which applies Luong attention (https://arxiv.org/abs/1508.04025).
    """
    def __init__(self, n_words, hidden_size, num_layers, drop_p):
        """
        Args:
            n_words (integer): Size of the target vocabulary.
            hidden_size (integer): Size of GRU cells.
            num_layers (integer): Number of GRU layers.
            drop_p (float): Probability to drop elements at Dropout layers.
        """
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embed = nn.Embedding(n_words, hidden_size)
        self.cell = MultiLayerGRUCell(2 * hidden_size,
                                      hidden_size,
                                      num_layers=num_layers,
                                      drop_p=drop_p)
        # The initial states are trainable vectors.
        self.init_h = torch.nn.Parameter(torch.randn([num_layers, 1, hidden_size]))
        self.init_y = torch.nn.Parameter(torch.randn([1, hidden_size]))

        self.attn_W = nn.Linear(2 * hidden_size, hidden_size)
        self.attn_U = nn.Linear(hidden_size, hidden_size)
        self.attn_v = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(3 * hidden_size, hidden_size)
        self.drop = nn.Dropout(drop_p)
        self.classifier = nn.Linear(hidden_size, n_words)

    def forward(self, encoder_states, ground_truths=None, beam_width=1):
        """
        The forwarding behavior depends on if ground-truths are provided.

        Args:
            encoder_states (PackedSequence): Packed output state vectors from the EncoderRNN.
            ground_truths (torch.LongTensor, [batch_size, padded_len_tgt]): Padded ground-truths.
            beam_width (integer): Beam Search width. Beam Search is equivalent to Greedy Search when beam_width=1.

        Returns:
            * When ground-truths are provided, it returns cross-entropy loss. Otherwise it returns predicted word IDs
            and the attention weights.
            loss (float): The cross-entropy loss to maximizing the probability of generating ground-truths.
            predictions (torch.FloatTensor, [batch_size, max_length]): The generated sentence.
            all_attn_weights (torch.FloatTensor, [batch_size, max_length, length_of_encoder_states]): A list contains
                attention alignment weights for the predictions.
        """
        states, states_lengths  = rnn_utils.pad_packed_sequence(
            encoder_states, batch_first=True)   # [batch_size, padded_len_src, 2 * hidden_size], [batch_size]
        batch_size = states.shape[0]
        h = self.init_h.repeat([1, batch_size, 1])   # [num_layers, batch_size, hidden_size]
        y = self.init_y.repeat([batch_size, 1])      # [batch_size, hidden_size]

        if ground_truths is None:
            if beam_width == 1:
                ## Greedy Search
                all_attn_weights = []
                predictions = [torch.full([batch_size], 3, dtype=torch.int64).cuda()]   # The first predicted word is always <s> (ID=3).
                # Unrolling the forward pass
                for time_step in range(100):   # Empirically set max_length=100
                    x = predictions[-1]                           # [batch_size]
                    x = self.embed(x)                             # [batch_size, hidden_size]
                    h = self.cell(torch.cat([y, x], dim=-1), h)   # [num_layers, batch_size, hidden_size]
                    attns, attn_weights = self.apply_attn(
                        states, states_lengths, h[-1])            # [batch_size, 2 * hidden_size], [batch_size, length_of_encoder_states]
                    y = torch.cat([attns, h[-1]], dim=-1)         # [batch_size, 3 * hidden_size]
                    y = F.relu(self.fc(y))                        # [batch_size, hidden_size]
                    all_attn_weights.append(attn_weights)
                    # Output
                    logits = self.classifier(y)                   # [batch_size, n_words]
                    samples = torch.argmax(logits, dim=-1)        # [batch_size]
                    predictions.append(samples)
                all_attn_weights = torch.stack(all_attn_weights, dim=1)   # [batch_size, max_length, length_of_encoder_states]
                predictions = torch.stack(predictions, dim=-1)    # [batch_size, max_length]
                return predictions, all_attn_weights
            else:
                ## Beam search
                assert batch_size == 1, ("Only Greedy Search (beam_width=1) supports batch size > 1.")
                beams = [{'h':h,
                          'y':y,
                          'preds': [torch.full([1], 3, dtype=torch.int64).cuda()],   # The first predicted word is always <s> (ID=3).
                          'attn_weights': [],
                          'scores':[]}]
                for time_step in range(100):   # Empirically set max_length=100
                    beams_update = []
                    for i in range(len(beams)):
                        b = beams[i]
                        if b['preds'][-1].item() == 2:   # Stop predicting if </s> (ID=2) has been sampled.
                            beams_update.append(b)
                            continue
                        else:
                            x = b['preds'][-1]   # [1]
                            x = self.embed(x)    # [1, hidden_size]
                            b['h'] = self.cell(torch.cat([b['y'], x], dim=-1), b['h'])   # [num_layers, 1, hidden_size]
                            attns, attn_weights = self.apply_attn(
                                states, states_lengths, b['h'][-1])       # [1, 2 * hidden_size], [1, length_of_encoder_states]
                            y = torch.cat([attns, b['h'][-1]], dim=-1)    # [1, 3 * hidden_size]
                            y = F.relu(self.fc(y))                        # [1, hidden_size]
                            b['y'] = y
                            b['attn_weights'] = b['attn_weights'] + [attn_weights]
                            # Output
                            scores = F.log_softmax(self.classifier(y), dim=-1)           # [1, n_words]
                            scores, samples = torch.topk(scores, k=beam_width, dim=-1)   # [1, beam_width], [1, beam_width]
                            for j in range(beam_width):
                                b_branch = b.copy()
                                b_branch['preds'] = b_branch['preds'] + [samples[:,j]]
                                b_branch['scores'] = b_branch['scores'] + [scores[:,j].item()]
                                beams_update.append(b_branch)
                    beams = beams_update
                    # Pick the best beams.
                    beams = sorted(beams, key=lambda b: np.mean(b['scores']), reverse=True)
                    beams = beams[:beam_width]
                b = beams[0]
                predictions = b['preds']                                  # list(LongTensor)
                predictions = torch.stack(predictions, dim=-1)            # [1, seq_length]
                all_attn_weights = b['attn_weights']                      # list(FloatTensor)
                all_attn_weights = torch.stack(all_attn_weights, dim=1)   # [1, seq_length, length_of_encoder_states]
                return predictions, all_attn_weights
        else:
            ## Compute loss
            xs = self.embed(ground_truths[:, :-1])   # [batch_size, padded_len_tgt, hidden_size]
            outputs = []
            # Unrolling the forward pass
            for time_step in range(xs.shape[1]):
                h = self.cell(torch.cat([y, xs[:,time_step]], dim=-1), h)   # [num_layers, batch_size, hidden_size]
                attns, _ = self.apply_attn(states, states_lengths, h[-1])   # [batch_size, 2 * hidden_size]
                y = torch.cat([attns, h[-1]], dim=-1)                       # [batch_size, 3 * hidden_size]
                y = F.relu(self.fc(y))                                      # [batch_size, hidden_size]
                outputs.append(y)

            # Output
            outputs = torch.stack(outputs, dim=1)   # [batch_size, padded_len_tgt, hidden_size]
            outputs = self.drop(outputs)
            outputs = self.classifier(outputs)      # [batch_size, padded_len_tgt, n_words]

            mask = ground_truths[:, 1:].gt(0)       # [batch_size, padded_len_tgt]
            loss = nn.CrossEntropyLoss()(outputs[mask], ground_truths[:, 1:][mask])
            return loss

    def apply_attn(self, source_states, source_lengths, target_states):
        """
        Apply attention.

        Args:
            source_states (torch.FloatTensor, [batch_size, padded_length_of_encoder_states, 2 * hidden_size]):
                The padded encoder output states.
            source_lengths (torch.LongTensor, [batch_size]): The length of encoder output states before padding.
            target_state (torch.FloatTensor, [batch_size, hidden_size]): The decoder output state (of previous time step).

        Returns:
            attns (torch.FloatTensor, [batch_size, hidden_size]):
                The attention result (weighted sum of Encoder output states).
            attn_weights (torch.FloatTensor, [batch_size, padded_length_of_encoder_states]): The attention alignment weights.
        """
        # A two-layer network used for project every pair of [source_state, target_state].
        attns = self.attn_W(source_states) + self.attn_U(target_states).unsqueeze(1)   # [batch_size, padded_len_src, hidden_size]
        attns = self.attn_v(F.relu(attns)).squeeze(2)                   # [batch_size, padded_len_src]

        # Create a mask with shape [batch_size, padded_len_src] to ignore the encoder states with <PAD> tokens.
        mask = torch.arange(attns.shape[1]).unsqueeze(0).repeat([attns.shape[0], 1]).ge(source_lengths.unsqueeze(1))
        attns = attns.masked_fill_(mask.cuda(), -float('inf'))          # [batch_size, padded_len_src]
        attns = F.softmax(attns, dim=-1)                                # [batch_size, padded_len_src]
        attn_weights = attns.clone()
        attns = torch.sum(source_states * attns.unsqueeze(-1), dim=1)   # [batch_size, 2 * hidden_size]
        return attns, attn_weights


class Seq2Seq(nn.Module):
    """
    Sequence-to-sequence model at high-level view. It is made up of an EncoderRNN module and a DecoderRNN module.
    """
    def __init__(self, target_size, hidden_size, encoder_layers, decoder_layers, drop_p=0., use_bn=True):
        """
        Args:
            target_size (integer): Target vocabulary size.
            hidden_size (integer): Size of GRU cells.
            encoder_layers (integer): EncoderRNN layers.
            decoder_layers (integer): DecoderRNN layers.
            drop_p (float): Probability to drop elements at Dropout layers.
            use_bn (bool): Whether to insert BatchNorm in EncoderRNN.
        """
        super(Seq2Seq, self).__init__()

        self.encoder = EncoderRNN(hidden_size, encoder_layers, use_bn)
        self.decoder = DecoderRNN(target_size, hidden_size, decoder_layers, drop_p)

    def forward(self, xs, xlens, ys=None, beam_width=1):
        """
        The forwarding behavior depends on if ground-truths are provided.

        Args:
            xs (torch.LongTensor, [batch_size, seq_length, dim_features]): A mini-batch of FBANK features.
            xlens (torch.LongTensor, [batch_size]): Sequence lengths before padding.
            ys (torch.LongTensor, [batch_size, padded_length_of_target_sentences]): Padded ground-truths.
            beam_width (integer): Beam Search width. Beam Search is equivalent to Greedy Search when beam_width=1.

        Returns:
            * When ground-truths are provided, it returns cross-entropy loss. Otherwise it returns predicted word IDs
            and the attention weights.
            loss (float): The cross-entropy loss to maximizing the probability of generating ground-truth.
            predictions (torch.FloatTensor, [batch_size, max_length]): The generated sentence.
            attn_weights (torch.FloatTensor, [batch_size, max_length, length_of_encoder_states]): A list contains
                attention alignment weights for the predictions.
        """
        if ys is None:
            predictions, attn_weights = self.decoder(self.encoder(xs, xlens), beam_width=beam_width)
            return predictions, attn_weights
        else:
            loss = self.decoder(self.encoder(xs, xlens), ys)
            return loss
