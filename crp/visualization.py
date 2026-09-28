from typing import List, Union, Dict, Tuple, Callable
import gc
import json
import re
import warnings
import torch
import numpy as np
import math
from collections.abc import Iterable
import concurrent.futures
import functools
import itertools
import inspect
from tqdm import tqdm
from zennit.composites import NameMapComposite, Composite
from crp.attribution import CondAttribution
from crp.maximization import Maximization
from crp.concepts import Concept, qwen_visual_token_counts
from crp.statistics import Statistics
from crp.hooks import FeatVisHook
from crp.helper import load_maximization, load_statistics, load_stat_targets
from crp.image import vis_img_heatmap, vis_opaque_img
from crp.cache import Cache

class QwenFeatureVisualization:
    def __init__(
            self, attribution: CondAttribution, dataset, layer_map: Dict[str, Concept], processor=None,
            max_target="sum", abs_norm=True, path="FeatureVisualization", device=None, cache: Cache = None,
            max_new_tokens=20, include_eos=False):

        self.dataset = dataset
        self.layer_map = layer_map
        self.processor = processor
        self.max_new_tokens = int(max_new_tokens)
        self.include_eos = bool(include_eos)
        self.prediction_records = []

        self.attribution = attribution

        self.device = attribution.device if device is None else device

        self.RelMax = Maximization("relevance", max_target, abs_norm, path)
        self.ActMax = Maximization("activation", max_target, abs_norm, path)

        self.RelStats = Statistics("relevance", max_target, abs_norm, path)
        self.ActStats = Statistics("activation", max_target, abs_norm, path)

        self.Cache = cache

    def run(self, data_start, data_end, composite: Composite = None, batch_size=32, checkpoint=500, on_device=None):
        """

        :param data_start: 0 | any other data point
        :param data_end: len(dataset) | any other data point
        :param composite: composite for LRP/AttnLRP
        :param batch_size:
        :param checkpoint:
        :param on_device:
        :return:
        """
        saved_checkpoints = self.run_distributed(data_start, data_end, composite, batch_size, checkpoint, on_device)

        saved_files = self.collect_results(saved_checkpoints)

        return saved_files

    def run_distributed(self, data_start, data_end, composite: Composite = None, batch_size=32, checkpoint=500,
                        on_device=None):
        """
        max batch_size = max(multi_targets) * data_batch
        data_end: exclusively counted
        """

        self.saved_checkpoints = {"r_max": [], "a_max": [], "r_stats": [], "a_stats": []}
        last_checkpoint = 0

        n_samples = data_end - data_start
        if n_samples <= 0:
            raise ValueError("data_end must be greater than data_start")
        if batch_size <= 0 or checkpoint <= 0:
            raise ValueError("batch_size and checkpoint must be positive")
        # samples: array(int)
        samples = np.arange(start=data_start, stop=data_end) # Ex.: np.arange(0,3) --> array([0, 1, 2])

        if n_samples > batch_size:
            batches = math.ceil(n_samples / batch_size)
        else:
            batches = 1
            batch_size = n_samples

        # feature visualization is performed inside forward and backward hook of layers
        name_map, dict_inputs = [], {}

        for l_name, concept in self.layer_map.items():
            hook = FeatVisHook(self, concept, l_name, dict_inputs, on_device)
            name_map.append(([l_name], hook))
        fv_composite = NameMapComposite(name_map)  #  maps module types to LRP rules, so that when this module is encountered, a corresponding hook is registered

        # Keep the feature-visualization hooks active for the scan.  The LRP
        # composite is deliberately registered only for one attribution at a
        # time below: Zennit hooks retain their latest forward tensors, so a
        # scan-wide registration makes GPU memory grow after every sample.
        fv_composite.register(self.attribution.model)   # stores activation(relevance) scores

        pbar = tqdm(total=batches, dynamic_ncols=True)

        for b in range(batches):
            pbar.update(1)

            sample_indices = samples[b * batch_size: (b + 1) * batch_size]
            inputs, true_answers = self.get_data_concurrently(sample_indices)

            # Generate the model's actual answer. A second forward pass below
            # reconstructs logits for the union of all generated answer tokens.
            prompt_width = inputs.input_ids.shape[1]
            # Feature-visualization hooks must observe only the attributed
            # recomputation, not each autoregressive generation step.
            fv_composite.remove()
            try:
                with torch.no_grad():
                    output_ids = self.attribution.model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        pixel_values=inputs.pixel_values,
                        image_grid_thw=inputs.image_grid_thw,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
            finally:
                fv_composite.register(self.attribution.model)

            generated_width = output_ids.shape[1] - prompt_width
            if generated_width <= 0:
                raise RuntimeError("Qwen generated no answer tokens")
            generated_ids = output_ids[:, prompt_width:]
            recompute_ids = output_ids[:, :-1]
            inputs["input_ids"] = recompute_ids
            inputs["attention_mask"] = torch.ones_like(recompute_ids)
            inputs["inputs_embeds"] = (
                self.attribution.model.get_input_embeddings()(recompute_ids)
                .detach()
                .requires_grad_(True)
            )

            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            pad_id = getattr(tokenizer, "pad_token_id", None)
            eos_id = getattr(tokenizer, "eos_token_id", None)
            conditions, statistic_targets, decoded_token_lists = [], [], []
            for row in range(generated_ids.shape[0]):
                token_ids, token_conditions = [], []
                for position, token in enumerate(generated_ids[row].tolist()):
                    token = int(token)
                    if pad_id is not None and token == pad_id:
                        continue
                    if eos_id is not None and token == eos_id and not self.include_eos:
                        continue
                    token_ids.append(token)
                    token_conditions.append((position, token))
                if not token_conditions:
                    token = int(generated_ids[row, 0])
                    token_ids, token_conditions = [token], [(0, token)]
                conditions.append({
                    self.attribution.MODEL_OUTPUT_NAME: token_conditions
                })
                statistic_targets.append(token_ids[0])
                decoded_token_lists.append(token_ids)

            predictions = self.processor.batch_decode(
                decoded_token_lists,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for dataset_index, truth, prediction in zip(
                sample_indices, true_answers, predictions
            ):
                def normalize(value):
                    value = re.sub(r"[^\w\s-]", "", str(value).casefold())
                    return re.sub(r"\s+", " ", value).strip()
                self.prediction_records.append({
                    "dataset_index": int(dataset_index),
                    "true_answer": str(truth),
                    "prediction": str(prediction).strip(),
                    "exact_match": normalize(truth) == normalize(prediction),
                })

            targets = np.asarray(statistic_targets, dtype=np.int64)
            sample_indices = np.asarray(sample_indices)
            # dict_inputs is linked to FeatHooks
            dict_inputs["input_ids"] = inputs.input_ids
            dict_inputs["sample_indices"] = sample_indices
            dict_inputs["targets"] = targets
            additional_forward_kwargs = {
              "attention_mask": inputs.attention_mask,
              "image_grid_thw": inputs.image_grid_thw,
              "spatial_merge_size": self._spatial_merge_size(),
              "logits_to_keep": generated_width,
              }
            dict_inputs["additional_forward_kwargs"] = additional_forward_kwargs


            # composites are already registered before
            attr = self.attribution(
                inputs,  # input is a tensor or a tuple of tensors.
                conditions,
                # Attribution registers the LRP composite in a context manager
                # and removes it immediately afterwards.  This releases the
                # tensors stored by Zennit's BasicHooks between samples.
                composite=composite,
                record_layer=list(self.layer_map.keys()),
                additional_forward_kwargs=additional_forward_kwargs,
            )

            # Hook callbacks have already consumed the activations and
            # relevances. Do not retain the attribution result or input graph.
            del attr, inputs, conditions, additional_forward_kwargs

            if torch.cuda.is_available() and (b + 1) % 25 == 0:
                gc.collect()
                torch.cuda.empty_cache()


            # self.attribution((inputs.pixel_values, inputs.input_embeds), conditions, None, exclude_parallel=False,
            #                 additional_forward_kwargs=additional_forward_kwargs)

            if b % checkpoint == checkpoint - 1:
                self._save_results((last_checkpoint, b + 1))
                last_checkpoint = b + 1

        # TODO: what happens if result arrays are empty?
        self._save_results((last_checkpoint, b + 1))

        fv_composite.remove()

        pbar.close()

        prediction_path = self.RelMax.PATH.parent / "generated_predictions.jsonl"
        with prediction_path.open("w", encoding="utf-8") as stream:
            for record in self.prediction_records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        correct = sum(record["exact_match"] for record in self.prediction_records)
        total = len(self.prediction_records)
        accuracy = correct / total if total else 0.0
        print(
            f"Generated-answer exact-match accuracy: {correct}/{total} "
            f"({accuracy:.2%}). Saved to {prediction_path}"
        )

        return self.saved_checkpoints

    def get_data_concurrently(self, indices: Union[List, np.ndarray, torch.Tensor]):
        """
        Converts images to the input format required for Qwen (BatchFeature objects); extracts targets (ids)
        :param indices: indices of the dataset images that will be input to the model
        :return: inputs, targets
        """
        images, questions, answers = zip(*[self.dataset[int(i)] for i in indices])

        # process images in batches
        prompts = [
            [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Answer in ONE word."}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                },
            ]
            for image, question in zip(images, questions)
        ]

        texts = [
            self.processor.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]

        inputs = self.processor(
            text=texts,
            images=list(images),
            padding=True,
            return_tensors="pt",
        ).to(self.device)  # --> class 'transformers.feature_extraction_utils.BatchFeature'

        # The ground-truth answers are retained only for later correctness
        # evaluation. CRP relevance is initialized from Qwen's generated
        # multi-token answer inside run_distributed.
        inputs.to(self.attribution.model.device)
        inputs.pixel_values.requires_grad_(True)

        return inputs, list(answers)

    def _first_answer_token_id(self, answer):
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if tokenizer is None:
            raise ValueError("A Qwen processor/tokenizer is required to encode targets")
        encoded = tokenizer(str(answer), add_special_tokens=False)
        token_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        if not token_ids:
            raise ValueError(f"Answer {answer!r} does not produce a tokenizer token")
        return int(token_ids[0])

    def _spatial_merge_size(self):
        config = getattr(self.attribution.model, "config", None)
        vision_config = getattr(config, "vision_config", None)
        return int(getattr(vision_config, "spatial_merge_size", 1))

    def _qwen_pixel_relevance_map(self, pixel_values, relevance, grid):
        """Convert flattened Qwen patch relevance to a dense pixel heatmap.

        Qwen stores an image as rows of flattened
        ``[channels, temporal_patch, patch_h, patch_w]`` patches.  Patch rows
        are ordered in spatial-merge groups rather than simple raster order.
        This method first applies the original CRP input-times-relevance rule,
        then reverses both flattening and merge-group ordering.
        """
        config = getattr(self.attribution.model, "config", None)
        vision_config = getattr(config, "vision_config", None)
        patch_size = int(getattr(vision_config, "patch_size", 14))
        temporal_patch_size = int(
            getattr(vision_config, "temporal_patch_size", 2)
        )
        merge_size = self._spatial_merge_size()
        t, h, w = map(int, grid.tolist())

        features_per_channel = temporal_patch_size * patch_size * patch_size
        if pixel_values.shape[1] % features_per_channel:
            raise ValueError(
                f"Qwen pixel feature width {pixel_values.shape[1]} is not "
                f"divisible by temporal_patch_size*patch_size^2 "
                f"({features_per_channel})"
            )
        channels = pixel_values.shape[1] // features_per_channel
        expected_patches = t * h * w
        if pixel_values.shape != relevance.shape:
            raise ValueError(
                f"Input and relevance shapes differ: {pixel_values.shape} vs "
                f"{relevance.shape}"
            )
        if pixel_values.shape[0] != expected_patches:
            raise ValueError(
                f"Grid describes {expected_patches} patches, received "
                f"{pixel_values.shape[0]}"
            )
        if h % merge_size or w % merge_size:
            raise ValueError(
                f"Patch grid {(h, w)} is not divisible by spatial merge "
                f"size {merge_size}"
            )

        # The attribution object exposes the input relevance in .grad. Match
        # the original CRP input heatmap definition before aggregating RGB and
        # the duplicated temporal frames used for still images.
        pixel_relevance = (pixel_values * relevance).reshape(
            expected_patches,
            channels,
            temporal_patch_size,
            patch_size,
            patch_size,
        ).sum(dim=(1, 2))

        # Reverse Qwen2.5-VLImageProcessor's patch order:
        # [t, h//m, w//m, m_h, m_w, patch_h, patch_w]
        dense = pixel_relevance.reshape(
            t,
            h // merge_size,
            w // merge_size,
            merge_size,
            merge_size,
            patch_size,
            patch_size,
        ).permute(0, 1, 3, 5, 2, 4, 6).contiguous().reshape(
            t,
            h * patch_size,
            w * patch_size,
        )

        # Still images normally have t=1. Summing also gives a meaningful map
        # for multi-frame inputs while conserving total signed relevance.
        return dense.sum(dim=0)

    def aggregate_qwen_vision_by_image(
        self,
        x,
        additional_forward_kwargs,
        mode="sum",
    ):
        """
        x:
            [total_visual_tokens, channels]

        returns:
            [batch, channels]
        """

        grid = additional_forward_kwargs["image_grid_thw"]

        token_counts = qwen_visual_token_counts(
            x.shape[0], additional_forward_kwargs
        )

        chunks = torch.split(
            x,
            token_counts.tolist(),
            dim=0,
        )

        if mode == "sum":
            return torch.stack(
                [chunk.sum(dim=0) for chunk in chunks],
                dim=0,
            )

        elif mode == "mean":
            return torch.stack(
                [chunk.mean(dim=0) for chunk in chunks],
                dim=0,
            )

        elif mode == "max":
            return torch.stack(
                [chunk.amax(dim=0) for chunk in chunks],
                dim=0,
            )

        raise ValueError(f"Unknown aggregation mode: {mode}")

    def _select_packed_images(self, x, additional_forward_kwargs, indices):
        """Select complete images from a packed visual-token tensor."""
        counts = qwen_visual_token_counts(x.shape[0], additional_forward_kwargs)
        chunks = torch.split(x, counts.tolist(), dim=0)
        selected = [chunks[int(index)] for index in indices]
        kwargs = dict(additional_forward_kwargs)
        grid = additional_forward_kwargs["image_grid_thw"]
        index_tensor = torch.as_tensor(indices, device=grid.device, dtype=torch.long)
        kwargs["image_grid_thw"] = grid.index_select(0, index_tensor)
        return torch.cat(selected, dim=0), kwargs
    
    @torch.no_grad()
    def analyze_relevance(
        self,
        rel,
        layer_name,
        concept,
        data_indices,
        targets,
        additional_forward_kwargs,
    ):
        #print("raw relevance:", rel.shape)

        d_c_sorted, rel_c_sorted, rf_c_sorted, t_c_sorted = (
            self.RelMax.analyze_layer(
                torch.abs(rel),
                concept,
                layer_name,
                data_indices,
                targets,
                additional_forward_kwargs,
            )
        )

        self.RelStats.analyze_layer(
            d_c_sorted, rel_c_sorted, rf_c_sorted, t_c_sorted, layer_name
        )

        return d_c_sorted, rel_c_sorted, rf_c_sorted, t_c_sorted

    @torch.no_grad()
    def analyze_activation(
        self,
        act,
        layer_name,
        concept,
        data_indices,
        targets,
        additional_forward_kwargs,
    ):
        #print("raw activation:", act.shape)

        # Activations do not depend on the duplicated answer target. Keep one
        # complete packed token span for each original image, while retaining
        # the token axis needed to calculate receptive-field locations.
        unique_indices = np.unique(
            data_indices,
            return_index=True,
        )[1]

        if act.shape[0] != len(data_indices):
            act, additional_forward_kwargs = self._select_packed_images(
                act, additional_forward_kwargs, unique_indices
            )
        else:
            index_tensor = torch.as_tensor(
                unique_indices, device=act.device, dtype=torch.long
            )
            act = act.index_select(0, index_tensor)
            additional_forward_kwargs = dict(additional_forward_kwargs)
            grid = additional_forward_kwargs["image_grid_thw"]
            grid_indices = index_tensor.to(grid.device)
            additional_forward_kwargs["image_grid_thw"] = grid.index_select(
                0, grid_indices
            )

        data_indices = data_indices[unique_indices]
        targets = targets[unique_indices]

        #print("unique indices:", len(unique_indices))
        #print("analyze acts:", act.shape)

        d_c_sorted, act_c_sorted, rf_c_sorted, t_c_sorted = (
            self.ActMax.analyze_layer(
                act,
                concept,
                layer_name,
                data_indices,
                targets,
                additional_forward_kwargs,
            )
        )

        self.ActStats.analyze_layer(
            d_c_sorted,
            act_c_sorted,
            rf_c_sorted,
            t_c_sorted,
            layer_name,
        )

    def _save_results(self, d_index=None):

        self.saved_checkpoints["r_max"].extend(self.RelMax._save_results(d_index))
        self.saved_checkpoints["a_max"].extend(self.ActMax._save_results(d_index))
        self.saved_checkpoints["r_stats"].extend(self.RelStats._save_results(d_index))
        self.saved_checkpoints["a_stats"].extend(self.ActStats._save_results(d_index))

    def collect_results(self, checkpoints: Dict[str, List[str]], d_index: Tuple[int, int] = None):

        saved_files = {}

        saved_files["r_max"] = self.RelMax.collect_results(checkpoints["r_max"], d_index)
        saved_files["a_max"] = self.ActMax.collect_results(checkpoints["a_max"], d_index)
        saved_files["r_stats"] = self.RelStats.collect_results(checkpoints["r_stats"], d_index)
        saved_files["a_stats"] = self.ActStats.collect_results(checkpoints["a_stats"], d_index)

        return saved_files

    def cache_reference(func):
        """
        Decorator for get_max_reference and get_stats_reference. If a crp.cache object is supplied to the FeatureVisualization object,
        reference samples are cached i.e. saved after computing a visualization with a 'plot_fn' (argument of get_max_reference) or
        loaded from the disk if available.
        """

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            """
            Parameters:
            -----------
            overwrite: boolean
                If set to True, already computed reference samples are computed again (overwritten).
            """

            overwrite = kwargs.pop("overwrite", False)
            args_f = inspect.getcallargs(func, self, *args, **kwargs)
            plot_fn = args_f["plot_fn"]

            if self.Cache is None or plot_fn is None:
                return func(**args_f)

            r_range, mode, l_name, rf, composite = args_f["r_range"], args_f["mode"], args_f["layer_name"], args_f[
                "rf"], args_f["composite"]
            f_name, plot_name = func.__name__, plot_fn.__name__
            if f_name == "get_max_reference":
                indices = args_f["concept_ids"]
            else:
                indices = [f'{args_f["concept_id"]}:{i}' for i in args_f["targets"]]

            if overwrite:
                not_found = {id: r_range for id in indices}
                ref_c = {}
            else:
                ref_c, not_found = self.Cache.load(indices, l_name, mode, r_range, composite, rf, f_name, plot_name)

            if len(not_found):

                for id in not_found:

                    args_f["r_range"] = not_found[id]

                    if f_name == "get_max_reference":
                        args_f["concept_ids"] = id
                        ref_c_left = func(**args_f)
                    elif f_name == "get_stats_reference":
                        args_f["targets"] = int(id.split(":")[-1])
                        ref_c_left = func(**args_f)
                    else:
                        raise ValueError(
                            "Only the methods 'get_max_reference' and 'get_stats_reference' can be decorated.")

                    self.Cache.save(ref_c_left, l_name, mode, not_found[id], composite, rf, f_name, plot_name)

                    ref_c = self.Cache.extend_dict(ref_c, ref_c_left)

            return ref_c

        return wrapper

    @cache_reference
    def get_max_reference(
            self, concept_ids: Union[int, list], layer_name: str, mode="relevance", r_range: Tuple[int, int] = (0, 8),
            attribute=False,
            rf=False, plot_fn=vis_img_heatmap, batch_size=32) -> Dict:
        """
        Retrieve reference samples for a list of concepts in a layer. Relevance and Activation Maximization
        are available if FeatureVisualization was computed for the mode. In addition, conditional heatmaps can be computed on reference samples.
        If the crp.concept class (supplied to the FeatureVisualization layer_map) implements masking for a single neuron in the 'mask_rf' method,
        the reference samples and heatmaps can be cropped using the receptive field of the most relevant or active neuron.

        Parameters:
        ----------
        concept_ids: int or list
        layer_name: str
        mode: "relevance" or "activation"
            Relevance or Activation Maximization
        r_range: Tuple(int, int)
            Range of N-top reference samples. For example, (3, 7) corresponds to the Top-3 to -6 samples.
            Argument must be a closed set i.e. second element of tuple > first element.
        composite: zennit.composites or None
            If set, compute conditional heatmaps on reference samples. `composite` is used for the CondAttribution object.
        rf: boolean
            If True, compute the CRP heatmap for the most relevant/most activating neuron only to restrict the conditonal heatmap
            on the receptive field.
        plot_fn: callable function with signature (samples: torch.Tensor, heatmaps: torch.Tensor, rf: boolean) or None
            Draws reference images. The function receives as input the samples used for computing heatmaps before preprocessing
            with self.preprocess_data and the final heatmaps after computation. In addition, the boolean flag 'rf' is passed to it.
            The return value of the function should correspond to the Cache supplied to the FeatureVisualization object (if available).
            If None, the raw tensors are returned.
        batch_size: int
            If heatmap is True, describes maximal batch size of samples to compute for conditional heatmaps.

        Returns:
        -------
        ref_c: dictionary.
            Key values correspond to channel index and values are reference samples. The values depend on the implementation of
            the 'plot_fn'.
        """

        ref_c = {}
        if not isinstance(concept_ids, Iterable):
            concept_ids = [concept_ids]
        if mode == "relevance":
            d_c_sorted, _, rf_c_sorted = load_maximization(self.RelMax.PATH, layer_name)
        elif mode == "activation":
            d_c_sorted, _, rf_c_sorted = load_maximization(self.ActMax.PATH, layer_name)
        else:
            raise ValueError("`mode` must be `relevance` or `activation`")

        if rf and not attribute:
            warnings.warn("The receptive field is only computed, if you set `attribute`.")

        for c_id in concept_ids:
            d_indices = d_c_sorted[r_range[0]:r_range[1], c_id]
            n_indices = rf_c_sorted[r_range[0]:r_range[1], c_id]

            ref_c[c_id] = self._load_ref_and_attribution(d_indices, c_id, n_indices, layer_name, attribute, rf, plot_fn,
                                                         batch_size)

        return ref_c

    @cache_reference
    def get_stats_reference(self, concept_ids: Union[int, list], layer_name: str, targets: Union[int, list],
                            mode="relevance", r_range: Tuple[int, int] = (0, 8),
                            attribute=False, rf=False, plot_fn=vis_img_heatmap, batch_size=32):
        """
        Retreive reference samples for a single concept in a layer wrt. different explanation targets i.e. returns the reference samples
        that are computed by self.compute_stats. Relevance and Activation are availble if FeatureVisualization was computed for the statitics mode.
        In addition, conditional heatmaps can be computed on reference samples. If the crp.concept class (supplied to the FeatureVisualization layer_map)
        implements masking for a single neuron in the 'mask_rf' method, the reference samples and heatmaps can be cropped using the receptive field of
        the most relevant or active neuron.

        Parameters:
        ----------
        concept_ids: int or list
        layer_name: str
        mode: "relevance" or "activation"
            Relevance or Activation Maximization
        r_range: Tuple(int, int)
            Range of N-top reference samples. For example, (3, 7) corresponds to the Top-3 to -6 samples.
            Argument must be a closed set i.e. second element of tuple > first element.
        attribute: boolean
            If set, compute conditional heatmaps on reference samples.
        rf: boolean
            If True, compute the CRP heatmap for the most relevant/most activating neuron only to restrict the conditonal heatmap
            on the receptive field.
        plot_fn: callable function with signature (samples: torch.Tensor, heatmaps: torch.Tensor, rf: boolean)
            Draws reference images. The function receives as input the samples used for computing heatmaps before preprocessing
            with self.preprocess and the final heatmaps after computation. In addition, the boolean flag 'rf' is passed to it.
            The return value of the function should correspond to the Cache supplied to the FeatureVisualization object (if available).
            If None, the raw tensors are returned.
        batch_size: int
            If heatmap is True, describes maximal batch size of samples to compute for conditional heatmaps.

        Returns:
        -------
        ref_t: dictionary.
            Key values correspond to target indices and values are reference samples. The values depend on the implementation of
            the 'plot_fn'.
        """

        ref_t = {}
        if not isinstance(targets, Iterable):
            targets = [targets]
        if not isinstance(concept_ids, Iterable):
            concept_ids = [concept_ids]
        if mode == "relevance":
            path = self.RelStats.PATH
        elif mode == "activation":
            path = self.ActStats.PATH
        else:
            raise ValueError("`mode` must be `relevance` or `activation`")

        if rf and not attribute:
            warnings.warn("The receptive field is only computed, if you set `attribute`.")

        for concept_id in concept_ids:
            for t in targets:
                d_c_sorted, _, rf_c_sorted = load_statistics(path, layer_name, t)
                d_indices = d_c_sorted[r_range[0]:r_range[1], concept_id]
                n_indices = rf_c_sorted[r_range[0]:r_range[1], concept_id]

                ref_t[f"{concept_id}:{t}"] = self._load_ref_and_attribution(d_indices, concept_id, n_indices,
                                                                            layer_name, attribute, rf, plot_fn,
                                                                            batch_size)

        return ref_t

    def _load_ref_and_attribution(self, d_indices, c_id, n_indices, layer_name, attribute, rf, plot_fn, batch_size):

        inputs, _ = self.get_data_concurrently(d_indices)
        if attribute:
            heatmaps = self._attribution_on_reference(inputs, c_id, layer_name, None, rf, n_indices, batch_size)
            if callable(plot_fn):
                return plot_fn(inputs.pixel_values, heatmaps[0], rf)
            else:
                return inputs, heatmaps

        else:
            return inputs

    def _attribution_on_reference(
      self,
      inputs,
      concept_id: int,
      layer_name: str,
      composite,
      rf=False,
      neuron_ids=None,
      batch_size=32,
  ):

      neuron_ids = [] if neuron_ids is None else neuron_ids
      n_samples = len(inputs.input_ids)

      if n_samples > batch_size:
          batches = math.ceil(n_samples / batch_size)
      else:
          batches = 1
          batch_size = n_samples

      if rf and len(neuron_ids) != n_samples:
          raise ValueError(
              "length of 'neuron_ids' must be equal "
              "to the length of 'inputs'"
          )

      # Number of raw Qwen vision patches per image
      patch_counts = inputs.image_grid_thw.prod(dim=1)

      # [0, patches_img0, patches_img0+patches_img1, ...]
      patch_offsets = torch.cat([
          torch.zeros(
              1,
              device=patch_counts.device,
              dtype=patch_counts.dtype,
          ),
          patch_counts.cumsum(dim=0),
      ])

      img_heatmaps = []
      txt_heatmaps = []

      for b in range(batches):

          sample_start = b * batch_size
          sample_end = min(
              (b + 1) * batch_size,
              n_samples,
          )

          # -----------------------------
          # Slice text/sample-level data
          # -----------------------------

          input_ids = inputs.input_ids[
              sample_start:sample_end
          ]

          attention_mask = inputs.attention_mask[
              sample_start:sample_end
          ]

          image_grid_thw = inputs.image_grid_thw[
              sample_start:sample_end
          ]

          # -----------------------------
          # Slice PATCH-level image data
          # -----------------------------

          patch_start = int(
              patch_offsets[sample_start].item()
          )

          patch_end = int(
              patch_offsets[sample_end].item()
          )

          pixel_values = inputs.pixel_values[
              patch_start:patch_end
          ]

          # Very useful sanity check
          expected_patches = (
              image_grid_thw
              .prod(dim=1)
              .sum()
              .item()
          )

          assert pixel_values.shape[0] == expected_patches, (
              f"Expected {expected_patches} patches, "
              f"got {pixel_values.shape[0]}"
          )

          # -----------------------------
          # Text embeddings
          # -----------------------------

          inputs_embeds = (
              self.attribution.model
              .get_input_embeddings()(input_ids)
              .detach()
              .requires_grad_(True)
          )

          if rf:
              batch_neuron_ids = neuron_ids[sample_start:sample_end]
              conditions = [
                  {
                      layer_name: {
                          int(concept_id): [int(neuron_index)]
                      }
                  }
                  for neuron_index in batch_neuron_ids
              ]
              mask_map = self.layer_map[layer_name].mask_rf
          else:
              conditions = [{layer_name: [concept_id]}]
              mask_map = self.layer_map[layer_name].mask

          attr = self.attribution(
              (pixel_values,),
              conditions,
              composite,
              mask_map=mask_map,
              start_layer=layer_name,
              on_device=self.device,
              exclude_parallel=False,
              additional_forward_kwargs={
                  "attention_mask": attention_mask,
                  "image_grid_thw": image_grid_thw,
                  "inputs_embeds": inputs_embeds,
                  "spatial_merge_size": self._spatial_merge_size(),
              },
              rf=rf,
          )

          input_relevance = attr.heatmap[0]

          # Split packed patches by image and reconstruct dense pixel-level
          # input-times-relevance heatmaps rather than one scalar per patch.
          batch_patch_counts = image_grid_thw.prod(dim=1).tolist()
          per_image_inputs = torch.split(pixel_values, batch_patch_counts)
          per_image_relevance = torch.split(input_relevance, batch_patch_counts)

          for image_input, image_relevance, grid in zip(
              per_image_inputs, per_image_relevance, image_grid_thw
          ):
              hm = self._qwen_pixel_relevance_map(
                  image_input, image_relevance, grid
              )
              img_heatmaps.append(hm.detach().cpu())

          if len(attr.heatmap) > 1:
              txt_heatmaps.extend(attr.heatmap[1].sum(-1).detach().cpu())

      if len(img_heatmaps) != n_samples:
          raise RuntimeError(
              f"Expected {n_samples} image heatmaps, got {len(img_heatmaps)}"
          )
      
      return (img_heatmaps, txt_heatmaps)


    def compute_stats(self, concept_id, layer_name: str, mode="relevance", top_N=5, mean_N=10, norm=False) -> Tuple[
        list, list]:
        """
        Computes statistics about the targets i.e. output classes for which the concept with index 'concept_id' in layer 'layer_name'
        is most relevant or most activated. Statistics must be computed before utilizing this method.

        Parameters:
        -----------
        concept_id: int
            Index of concept
        layer_name: str
        mode: str, 'relevance' or 'activation'
        top_N: int
            Returns the 'top_N' classes that most activate or are most relevant for the concept.
        mean_N: int
            Computes the importance of each target using the 'mean_N' top reference images for each target.
        norm: boolean
            If True, returns the mean relevance for each target normed.

        Returns:
        --------
        sorted_t, sorted_val as tuple
        sorted_t: list of most relevant targets
        sorted_val: list of respective mean relevance/activation values for each target
        """

        if mode == "relevance":
            path = self.RelStats.PATH
        elif mode == "activation":
            path = self.ActStats.PATH
        else:
            raise ValueError("`mode` must be `relevance` or `activation`")

        targets = load_stat_targets(path)

        rel_target = torch.zeros(len(targets))
        for i, t in enumerate(targets):
            _, rel_c_sorted, _ = load_statistics(path, layer_name, t)
            rel_target[i] = float(rel_c_sorted[:mean_N, concept_id].mean())

        args = torch.argsort(rel_target, descending=True)[:top_N]

        sorted_t = targets[args]
        sorted_val = rel_target[args]

        if norm:
            sorted_val = sorted_val / sorted_val[0]

        return sorted_t, sorted_val

    def _save_precomputed(self, s_tensor, h_tensor, index, plot_list, layer_name, mode, r_range, composite, rf, f_name):

        for plot_fn in plot_list:
            ref = {index: plot_fn(s_tensor, h_tensor, rf)}
            self.Cache.save(ref, layer_name, mode, r_range, composite, rf, f_name, plot_fn.__name__)

    def precompute_ref(self, layer_c_ind: Dict[str, List], composite: Composite, rf=True, stats=False, top_N=4,
                       mean_N=10, mode="relevance", r_range: Tuple[int, int] = (0, 8), plot_list=[vis_opaque_img],
                       batch_size=32):
        """
        Precomputes and saves all reference samples resulting from 'self.get_ref_samples' and 'self.get_stats_reference' for concepts supplied in 'layer_c_ind'.

        Parameters:
        -----------
        layer_c_ind: dict with str keys and list values
            Keys correspond to layer names and values to a list of all concept indices
        stats: boolean
            If True, precomputes reference samples of 'self.get_stats_reference'. Otherwise, only samples of 'self.get_ref_samples' are computed.
        plot_list: list of callable functions
            Functions to plot and save the images. The signature should correspond to the 'plot_fn' of 'get_max_reference'.

        REMAINING PARAMETERS: correspond to 'self.get_ref_samples' and 'self.get_stats_reference'
        """

        if self.Cache is None:
            raise ValueError(
                "You must supply a crp.Cache object to the 'FeatureVisualization' class to precompute reference images!")

        if composite is None:
            raise ValueError("You must supply a zennit.Composite object to precompute reference images!")

        for l_name in layer_c_ind:

            c_indices = layer_c_ind[l_name]
            #print("Layer:", l_name)
            pbar = tqdm(total=len(c_indices), dynamic_ncols=True)

            for c_id in c_indices:

                s_tensor, h_tensor = \
                    self.get_max_reference(c_id, l_name, mode, r_range, composite, rf, None, batch_size)[c_id]

                self._save_precomputed(s_tensor, h_tensor, c_id, plot_list, l_name, mode, r_range, composite, rf,
                                       "get_max_reference")

                if stats:
                    targets, _ = self.compute_stats(c_id, l_name, mode, top_N, mean_N)
                    for t in targets:
                        stat_index = f"{c_id}:{t}"
                        s_tensor, h_tensor = \
                            self.get_stats_reference(c_id, l_name, t, mode, r_range, composite, rf, None, batch_size)[
                                stat_index]
                        self._save_precomputed(s_tensor, h_tensor, stat_index, plot_list, l_name, mode, r_range,
                                               composite, rf, "get_stats_reference")

                pbar.update(1)

            pbar.close()

       
