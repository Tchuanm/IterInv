import torch
from diffusers.models.cross_attention import CrossAttention
from diffusers.models.attention import Attention


class MyKVProcessor:
    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        residual = hidden_states
        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj)

        if not attn.only_cross_attention:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # new bookkeeping to save the attn probs
        attn.attn_probs = attention_probs

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
        hidden_states = hidden_states + residual

        return hidden_states


class MyCrossAttnProcessor:
    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, H, W = hidden_states.shape        # 2,384,32,32
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size=batch_size)
        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1).transpose(1, 2)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)      # 2 1024 384   
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj)

        if not attn.only_cross_attention:       # cross + self
            _key = attn.to_k(hidden_states)
            _value = attn.to_v(hidden_states)
            _key = attn.head_to_batch_dim(_key)
            _value = attn.head_to_batch_dim(_value)
            key = torch.cat([encoder_hidden_states_key_proj, _key], dim=1)
            value = torch.cat([encoder_hidden_states_value_proj, _value], dim=1)
        else:       # only cross

            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # new bookkeeping to save the attn probs
        attn.attn_probs = attention_probs

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states.permute(0,2,1).reshape(batch_size, sequence_length, H, W)


"""
A function that prepares a U-Net model for training by enabling gradient computation 
for a specified set of parameters and setting the forward pass to be performed by a 
custom cross attention processor.

Parameters:
unet: A U-Net model.

Returns:
unet: The prepared U-Net model.
"""
def prep_unet(unet):
    # set the gradients for XA maps to be true
    for name, params in unet.named_parameters():
        # if 'attn2' in name:
        # flag = 'attentions.0' in name or 'attentions.2' in name # or 'attentions.3' in name
        # if flag:
        if 'attentions' in name:
            params.requires_grad = True
        else:
            params.requires_grad = False
    # replace the fwd function
    # count = 0
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        # if module_name == "CrossAttention":
        # flag = 'attentions.1' in name or 'attentions.2' in name or 'attentions.3' in name
        if module_name == 'Attention':
            module.set_processor(MyKVProcessor())
            # count += 1 
            # print('replace cross-attn process, count:', count)
    return unet
