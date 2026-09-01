import torch
import numpy as np
from typing import List, Dict
from sklearn.decomposition import FastICA


class Concept:
    """
    Abstract class that imlplements the core functionality for the attribution computation of concepts.
    """

    def mask(self, batch_id, concept_ids, layer_name, additional_forward_kwargs=None, rf=False):

        raise NotImplementedError("'Concept'class must be implemented!")

    def mask_rf(self, neuron_ids, layer_name, additional_forward_kwargs=None, rf=False):

        raise NotImplementedError("'Concept'class must be implemented!")

    def reference_sampling(self, relevance_or_activation, layer_name: str = None, max_target: str = "sum", abs_norm=True, additional_forward_kwargs=None):

        raise NotImplementedError("'Concept'class must be implemented!")

    def get_rf_indices(self, output_shape, layer_name):

        raise NotImplementedError("'Concept'class must be implemented!")

    def attribute(self, relevance, mask=None, layer_name: str = None, abs_norm=True):

        raise NotImplementedError("'Concept'class must be implemented!")


class ChannelConcept(Concept):
    """
    Concept Class for torch.nn.Conv2D and torch.nn.Linear layers
    """

    @staticmethod
    def mask(batch_id: int, concept_ids: List, layer_name=None, additional_forward_kwargs=None, rf=False):
        """
        Wrapper that generates a function thath modifies the gradient (replaced by zennit by attributions).

        Parameters:
        ----------
        batch_id: int
            Specifies the batch dimension in the torch.Tensor.
        concept_ids: list of integer values
            integer lists corresponding to channel indices.

        Returns:
        --------
        callable function that modifies the gradient
        """


        def mask_fct(grad):

            mask = torch.zeros_like(grad[batch_id])
            mask[concept_ids] = 1
            grad[batch_id] = grad[batch_id] * mask

            return grad

        def mask_fct_rf(grad):

            grad_shape = grad.shape
            grad = grad.view(*grad_shape[:2], -1)

            mask = torch.zeros_like(grad[batch_id])

            for concept_id in concept_ids:
                mask[concept_id, torch.argmax(torch.abs(grad[batch_id, concept_id]))] = 1

            grad[batch_id] = grad[batch_id] * mask
            
            return grad.view(grad_shape)

        if rf:
            return mask_fct_rf
        return mask_fct

    @staticmethod
    def mask_rf(batch_id: int, c_n_map: Dict[int, List], layer_name=None, additional_forward_kwargs=None, rf=False):
        """
        Wrapper that generates a function that modifies the gradient (replaced by zennit by attributions) for a single neuron.

        Parameters:
        ----------
        batch_id: int
            Specifies the batch dimension in the torch.Tensor.
        c_n_map: dist with int keys and list values
            Keys correspond to channel indices and values correspond to neuron indices.
            Neuron Indices are counted as if the 2D Channel has 1D dimension i.e. channel dimension [3, 20, 20] -> [3, 400],
            so that neuron indices range between 0 and 399.

        Returns:
        --------
        callable function that modifies the gradient
        """

        def mask_fct(grad):

            grad_shape = grad.shape
            grad = grad.view(*grad_shape[:2], -1)

            mask = torch.zeros_like(grad[batch_id])

            for channel in c_n_map:
            
                mask[channel, c_n_map[channel]] = 1

            grad[batch_id] = grad[batch_id] * mask
            return grad.view(grad_shape)

        return mask_fct

    def get_rf_indices(self, output_shape, layer_name=None):

        if len(output_shape) == 1:
            return [0]
        else:
            end = np.prod(output_shape[1:])
            return np.arange(0, end)

    def attribute(self, relevance, mask=None, layer_name: str = None, abs_norm=True):

        if isinstance(mask, torch.Tensor):
            relevance = relevance * mask

        rel_l = torch.sum(relevance.view(*relevance.shape[:2], -1), dim=-1)

        if abs_norm:
            rel_l = rel_l / (torch.abs(rel_l).sum(-1).view(-1, 1) + 1e-10)

        return rel_l

    def reference_sampling(self, relevance_or_activation, layer_name: str = None, max_target: str = "sum", abs_norm=True, additional_forward_kwargs=None):
        """
        Samples the most relevant/activated concepts for each sample in the batch. 
        Total channel relevance/activation can be defined as the sum or the maximum of the relevances/activations of all neurons in the channel.

        Parameters:
            relevance_or_activation: tensor [batch, tokens, embed_dim/channels]
            max_target: str. Either 'sum' or 'max'.
            abs_norm: bool. Whether the relevance/activations are normalized
        """

        # position of receptive field neuron
        rel_l = relevance_or_activation.view(*relevance_or_activation.shape[:2], -1)
        rf_neuron = torch.argmax(rel_l, dim=-1)

        # channel maximization target
        if max_target == "sum":
            rel_l = torch.sum(rel_l, dim=-1)

        elif max_target == "max":
            rel_l = torch.gather(rel_l, -1, rf_neuron.unsqueeze(-1)).squeeze(-1)

        else:
            raise ValueError("'max_target' supports only 'max' or 'sum'.")

        if abs_norm:
            rel_l = rel_l / (torch.abs(rel_l).sum(-1).view(-1, 1) + 1e-10)
        
        d_ch_sorted = torch.argsort(rel_l, dim=0, descending=True)
        rel_ch_sorted = torch.gather(rel_l, 0, d_ch_sorted)
        rf_ch_sorted = torch.gather(rf_neuron, 0, d_ch_sorted)

        return d_ch_sorted, rel_ch_sorted, rf_ch_sorted

class TransformerChannelConcept(Concept):
    """
    Concept Class for torch.nn.Linear transformer layers
    """

    @staticmethod
    def mask(batch_id: int, concept_ids: List, layer_name=None, additional_forward_kwargs=None, rf=False):
        """
        Wrapper that generates a function that modifies the gradient (replaced by zennit by attributions).

        Parameters:
        ----------
        batch_id: int
            Specifies the batch dimension in the torch.Tensor.
        concept_ids: list of integer values
            integer lists corresponding to neuron indices.

        Returns:
        --------
        callable function that modifies the gradient
        """

        def mask_fct(grad):
            # grad in transformer is [batch, tokens, embedding/channel], a channel is across tokens
            print(f"grad shape: {grad.shape}")

            mask = torch.zeros_like(grad[batch_id])
            mask[..., concept_ids] = 1
            grad[batch_id] = grad[batch_id] * mask

            return grad

        def mask_fct_rf(grad):

            mask = torch.zeros_like(grad[batch_id])

            for concept_id in concept_ids:
                mask[..., torch.argmax(torch.abs(grad[batch_id, ..., concept_id])), concept_id] = 1

            grad[batch_id] = grad[batch_id] * mask
            
            return grad

        if rf:
            return mask_fct_rf
        return mask_fct

    @staticmethod
    def mask_rf(batch_id: int, c_n_map: Dict[int, List], layer_name=None, additional_forward_kwargs=None, rf=False):
        """
        Wrapper that generates a function that modifies the gradient (replaced by zennit by attributions) for a single neuron.

        Parameters:
        ----------
        batch_id: int
            Specifies the batch dimension in the torch.Tensor.
        c_n_map: dist with int keys and list values
            Keys correspond to channel indices and values correspond to neuron indices.
            Neuron Indices are counted as if the 2D Channel has 1D dimension i.e. channel dimension [3, 20, 20] -> [3, 400],
            so that neuron indices range between 0 and 399.

        Returns:
        --------
        callable function that modifies the gradient
        """

        def mask_fct(grad):
            print(f"grad shape: {grad.shape}")

            mask = torch.zeros_like(grad[batch_id])
            
            for concept, neuron in c_n_map.items():
                mask[..., neuron, concept] = 1
                
            grad[batch_id] = grad[batch_id] * mask
            
            return grad

        return mask_fct
            
    def attribute(self, relevance, mask=None, layer_name: str = None, abs_norm=True):

        if isinstance(mask, torch.Tensor):
            relevance = relevance * mask

        print(f"relevance shape: {relevance.shape}")
        rel_l = relevance.sum(dim=-2)

        if abs_norm:
            rel_l = rel_l / (torch.abs(rel_l).sum(-1).view(-1, 1) + 1e-10)

        return rel_l

    def reference_sampling(self, relevance_or_activation, layer_name: str = None, max_target: str = "sum", abs_norm=True, additional_forward_kwargs=None):
        """
        Samples the most relevant/activated concepts for each sample in the batch. 
        Total channel relevance/activation can be defined as the sum or the maximum of the relevances/activations of all neurons in the channel.

        Parameters:
            relevance_or_activation: tensor [batch, tokens, embed_dim/channels]
            max_target: str. Either 'sum' or 'max'.
            abs_norm: bool. Whether the relevance/activations are normalized
        """
        print("rel/ act shape", relevance_or_activation.shape)
        
        # position of receptive field neuron
        rf_neuron = torch.argmax(relevance_or_activation, dim=-2)
        
        # channel maximization target --> sum neurons in channel across tokens
        if max_target == "sum":
            rel_l = torch.sum(relevance_or_activation, dim=-2)

        # if choosing only max target pick max neuron in channel across tokens
        elif max_target == "max":
            rel_l = torch.amax(relevance_or_activation, dim=-2)

        else:
            raise ValueError("'max_target' supports only 'max' or 'sum'.")
        
        if abs_norm:
            rel_l = rel_l / (torch.abs(rel_l).sum(-1, keepdim=True) + 1e-10)

        # sort in dataset index order
        d_ch_sorted = torch.argsort(rel_l, dim=0, descending=True)
        rel_ch_sorted = torch.gather(rel_l, dim=0, index=d_ch_sorted)
        rf_ch_sorted = torch.gather(rf_neuron, dim=0, index=d_ch_sorted)

        return d_ch_sorted, rel_ch_sorted, rf_ch_sorted


class ICAConcept(Concept):
    """
    Independent Component Analysis Concept Class for torch.nn.Linear transformer layers
    """

    def __init__(self, n_concepts: int, random_state: int = 42, device: str = "cuda"):
        self.n_concepts = n_concepts
        self.random_state = random_state
        self.concept_directions = {}
        self.components = {}
        self.mean = {}
        self.mixing = {}
        self.ica = FastICA(n_components=n_concepts, random_state=random_state)
        self.device = device

    def fit(self, layer_name: str, relevance_or_activation: torch.Tensor):
        self.ica.fit(relevance_or_activation)
        # normed concept directions
        self.concept_directions[layer_name] = torch.tensor(self.ica.components_ / np.linalg.norm(self.ica.components_, axis=-1, keepdims=True), device=self.device)
        self.components[layer_name] = torch.tensor(self.ica.components_, device=self.device).T
        self.mean[layer_name] = torch.tensor(self.ica.mean_, device=self.device)
        self.mixing[layer_name] = torch.tensor(self.ica.mixing_, device=self.device).T

    def _encode(self, layer_name, x):
        self._assert_fit(layer_name)
        return (x - self.mean[layer_name]) @ self.components[layer_name]

    def _decode(self, layer_name, z):
        self._assert_fit(layer_name)
        return (z @ self.mixing[layer_name]) + self.mean[layer_name]

    def _assert_fit(self, layer_name):
        if not layer_name in self.concept_directions:
            raise ValueError(f"Concept not yet fit to layer {layer_name} activations")

    def mask(self, batch_id: int, concept_ids: List, layer_name=None, additional_forward_kwargs=None, rf=False):
        """
        Wrapper that generates a function that modifies the gradient (replaced by zennit by attributions).

        Parameters:
        ----------
        batch_id: int
            Specifies the batch dimension in the torch.Tensor.
        concept_ids: list of integer values
            integer lists corresponding to neuron indices.

        Returns:
        --------
        callable function that modifies the gradient
        """

        def mask_fct(grad):
            # Use the concept directions as channel mask (i.e. across all token embeddings)
            # concept_mask = torch.sum([self.concept_directions[layer_name][concept_id] for concept_id in concept_ids], dim=0)
            # grad[batch_id] = grad[batch_id] * concept_mask.unsqueeze(0)

            concept_space = self._encode(layer_name, grad[batch_id])
            mask = torch.zeros_like(concept_space)
            mask[..., concept_ids] = 1
            concept_space = concept_space * mask
            grad[batch_id] = self._decode(layer_name, concept_space)

            return grad

        def mask_fct_rf(grad):

            concept_space = self._encode(layer_name, grad[batch_id])
            mask = torch.zeros_like(concept_space)

            for concept_id in concept_ids:
                mask[..., torch.argmax(torch.abs(concept_space[..., concept_id])), concept_id] = 1

            concept_space = concept_space * mask
            grad[batch_id] = self._decode(layer_name, concept_space)
            
            return grad

        if rf:
            return mask_fct_rf
        return mask_fct
            
    def attribute(self, relevance, mask=None, layer_name: str = None, abs_norm=True):

        if isinstance(mask, torch.Tensor):
            relevance = relevance * mask

        rel_l = self._encode(layer_name, relevance).sum(dim=-2)

        if abs_norm:
            rel_l = rel_l / (torch.abs(rel_l).sum(-1).view(-1, 1) + 1e-10)

        return rel_l

    def reference_sampling(self, relevance_or_activation, layer_name: str = None, max_target: str = "sum", abs_norm=True, additional_forward_kwargs=None):
        """
        Samples the most relevant/activated concepts for each sample in the batch. 
        Total channel relevance/activation can be defined as the sum or the maximum of the relevances/activations of all neurons in the channel.

        Parameters:
            relevance_or_activation: tensor [batch, tokens, embed_dim/channels]
            max_target: str. Either 'sum' or 'max'.
            abs_norm: bool. Whether the relevance/activations are normalized
        """
        
        concept_relevance_or_activation = self._encode(layer_name, relevance_or_activation)

        # position of receptive field neuron per concept channel
        rf_neurons = torch.argmax(concept_relevance_or_activation, dim=-2)
        
        # channel maximization target --> sum neurons in channel across tokens
        if max_target == "sum":
            rel_l = torch.sum(concept_relevance_or_activation, dim=-2)

        # if choosing only max target pick max neuron in channel across tokens
        elif max_target == "max":
            rel_l = torch.amax(concept_relevance_or_activation, dim=-2)

        else:
            raise ValueError("'max_target' supports only 'max' or 'sum'.")
        
        if abs_norm:
            rel_l = rel_l / (torch.abs(rel_l).sum(-1, keepdim=True) + 1e-10)

        # sort in dataset index order
        d_ch_sorted = torch.argsort(rel_l, dim=0, descending=True)
        rel_ch_sorted = torch.gather(rel_l, dim=0, index=d_ch_sorted)
        rf_ch_sorted = torch.gather(rf_neurons, dim=0, index=d_ch_sorted)

        return d_ch_sorted, rel_ch_sorted, rf_ch_sorted
