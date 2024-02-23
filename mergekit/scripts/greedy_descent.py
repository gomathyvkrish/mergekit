import click
import torch
import tqdm

from mergekit.architecture import get_architecture_info, WeightInfo
from mergekit.common import ModelReference
from mergekit.io.tasks import LoaderCache
from mergekit.io.tensor_writer import TensorWriter
from mergekit.options import MergeOptions, add_merge_options
from typing import Dict, Generic, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
import jax.numpy as jnp


@click.command("mergekit-rebasin-align")
@click.argument("model_path", type=str)
@click.option(
    "--target", "-t", required=True, type=str, help="Target model to align weights to"
)
@click.option("--out-path", "-o", required=True, type=str, help="Output model path")
@add_merge_options
def main(
    model_path: str,
    out_path: str,
    target: str,
    merge_options: MergeOptions,
):
    model = ModelReference.model_validate(model_path)
    target_model = ModelReference.model_validate(target)

    cache = LoaderCache()
    cache.lazy_unpickle = merge_options.lazy_unpickle
    cache.lora_cache_dir = merge_options.lora_merge_cache
    cache.hf_cache_dir = merge_options.transformers_cache
    cache.trust_remote_code = merge_options.trust_remote_code

    device = torch.device("cuda") if merge_options.cuda else torch.device("cpu")

    for m in tqdm.tqdm([model, target_model], desc="Fetching models"):
        cache.get(m)

    model_config = model.config(trust_remote_code=merge_options.trust_remote_code)
    model_arch_info = get_architecture_info(
        model.config(trust_remote_code=merge_options.trust_remote_code)
    )

    space_weights: Dict[str, Dict[ModelReference, List[WeightInfo]]] = {}

    def _get_tensor(model: ModelReference, name: str) -> torch.Tensor:
        """
        Helper function to get a tensor from a model
        """
        loader = cache.get(model)
        tensor = loader.get_tensor(name, device="cpu")
        return tensor
    
    def interpolate_weights(W_A, W_B, gamma=0.5):
        """
        Implements equation (1) in ZipIt paper to interpolate between two weight matrices.
        """
        return gamma * W_A + (1 - gamma) * W_B
    
    def permute_weights(W_A, W_B, P_i, P_i_prev, gamma=0.5):
        """
        Implements equation (2) in ZipIt paper to permute and interpolate between two weight matrices.
        """
        return gamma * W_A + (1 - gamma) * np.dot(np.dot(P_i, W_B), P_i_prev.T)

    def solve_lap(A, B):
        """Solves the linear assignment problem between matrices A and B."""
        if A.ndim == 1:
            A = A.reshape(-1, 1)
        if B.ndim == 1:
            B = B.reshape(-1, 1)
        C = -np.dot(A, B.T)  # Cost matrix
        row_ind, col_ind = linear_sum_assignment(C)
        P = np.zeros_like(C)
        P[row_ind, col_ind] = 1
        return P

    def add_weight_to_space(space_weights: Dict[str, Dict[str, List[WeightInfo]]], 
                            space_name: str, 
                            model_ref: ModelReference, 
                            weight_info: WeightInfo):
        if space_name not in space_weights:
            space_weights[space_name] = {}
        if model_ref not in space_weights[space_name]:
            space_weights[space_name][model_ref] = []
        space_weights[space_name][model_ref].append(weight_info)

    def print_tensors_in_spaces(space_weights: Dict[str, Dict[str, List[WeightInfo]]]):
        for space_name, models in space_weights.items():
            print(f"Space: {space_name}")
            for model_ref, weight_infos in models.items():
                print(f"  Model: {model_ref}")
                for weight in weight_infos:
                    print(f"    Weight Name: {weight.name}")
                    print(f"    Is Embed: {weight.is_embed}")
                    print(f"    Input Space: {weight.input_space}")
                    print(f"    Output Space: {weight.output_space}")
                    print(f"    Optional: {weight.optional}")

    def permutation_coordinate_descent(target_model, model, model_arch_info, model_config, gamma = 0.5):
        # Extract weights from models
        theta_A = []
        theta_B = []
        L = len(theta_A)
        perms = [np.eye(list(theta_A[l].shape)[-1]) for l in range(L-1)] 

        for weight_info in model_arch_info.all_weights(model_config):
            W_A = _get_tensor(target_model, weight_info.name).float().numpy()
            W_B = _get_tensor(model, weight_info.name).float().numpy()
            theta_A.append(W_A)
            theta_B.append(W_B)
        
        for weight_info in model_arch_info.all_weights(model_config):
            if weight_info.input_space:
                add_weight_to_space(space_weights, weight_info.input_space, target_model, weight_info)
                add_weight_to_space(space_weights, weight_info.input_space, model, weight_info)
            if weight_info.output_space:
                add_weight_to_space(space_weights, weight_info.output_space, target_model, weight_info)
                add_weight_to_space(space_weights, weight_info.output_space, model, weight_info)
        print_tensors_in_spaces(space_weights)

        while True:
            prev_perms = [P.copy() for P in perms]
            for l in np.random.permutation(len(theta_A) - 1):
                P_i_minus_1 = np.eye(theta_A[0].shape[0]) if l == 0 else perms[l - 1].T
                P_i = np.eye(theta_A[0].shape[0]) if l == len(theta_A) - 2 else perms[l + 1]
                
                W_star_simple = interpolate_weights(W_A, W_B, gamma)
                W_star_permuted = permute_weights(theta_A[l], theta_B[l], P_i, P_i_minus_1, gamma)
                
                new_weight_info = WeightInfo(name=weight_info.name, tensor=W_star_permuted)
                
                add_weight_to_space(space_weights, weight_info.output_space, target_model, new_weight_info)

                perms[l] = solve_lap(W_star_permuted)  # Update the permutation matrix

            # Check for convergence
            if all(np.array_equal(perms[l], prev_perms[l]) for l in range(len(theta_A) - 1)):
                break

        print_tensors_in_spaces(space_weights)
        return

    permutation_matrices = permutation_coordinate_descent(target_model, model, model_arch_info, model_config)
    # print(permutation_matrices)

    writer = TensorWriter(
        out_path,
        max_shard_size=merge_options.out_shard_size,
        safe_serialization=merge_options.safe_serialization,
    )
    
    # # Write the aligned model weights to the output path with `writer`
    # for layer_index, weight_info in enumerate(model_arch_info.all_weights(model_config)):
    #     if layer_index < len(permutation_matrices):
    #         # Retrieve the original weight tensor from model B
    #         original_tensor = _get_tensor(model, weight_info.name)

    #         # Apply the permutation matrix to the weight tensor
    #         permuted_tensor = torch.matmul(permutation_matrices[layer_index], original_tensor)

    #         # Save the permuted tensor using the TensorWriter
    #         writer.save_tensor(weight_info.name, permuted_tensor, clone=True)
    #     else:
    #         # For layers without a permutation matrix, just save the original tensor
    #         tensor = _get_tensor(model, weight_info.name)
    #         writer.save_tensor(weight_info.name, tensor, clone=True)

    # writer.flush_current_shard()  # Flush the last shard to ensure all tensors are written
    # writer.close()

if __name__ == "__main__":
    with torch.no_grad():
        main()
