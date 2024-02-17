import click
import torch
import tqdm

from mergekit.architecture import get_architecture_info
from mergekit.common import ModelReference
from mergekit.io.tasks import LoaderCache
from mergekit.io.tensor_writer import TensorWriter
from mergekit.options import MergeOptions, add_merge_options
import numpy as np
from scipy.optimize import linear_sum_assignment


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

    def _get_tensor(model: ModelReference, name: str) -> torch.Tensor:
        """
        Helper function to get a tensor from a model
        """
        loader = cache.get(model)
        tensor = loader.get_tensor(name, device="cpu")
        return tensor

    def solve_lap(A, B):
        """Solves the linear assignment problem between matrices A and B."""
        # if A.ndim == 1:
        #     A = A.reshape(-1, 1)
        # if B.ndim == 1:
        #     B = B.reshape(-1, 1)
        C = -np.dot(A, B.T)  # Cost matrix
        row_ind, col_ind = linear_sum_assignment(C)
        P = np.zeros_like(C)
        P[row_ind, col_ind] = 1
        return P

    def permutation_coordinate_descent(target_model, model, model_arch_info, model_config):
        # Extract weights from models
        theta_A = []
        theta_B = []
        for weight_info in model_arch_info.all_weights(model_config):
            tensor_a = _get_tensor(target_model, weight_info.name).float().numpy()
            tensor_b = _get_tensor(model, weight_info.name).float().numpy()
            theta_A.append(tensor_a)
            theta_B.append(tensor_b)

        L = len(theta_A)
        perms = [np.eye(theta_A[l].shape[0]) for l in range(L - 1)]  # Initialize perms as identity matrices

        # Repeat until convergence
        while True:
            prev_perms = [P.copy() for P in perms]
            for l in np.random.permutation(L - 1):
                if l > 0:
                    A = np.dot(theta_A[l], perms[l - 1].T)
                else:
                    A = theta_A[l]

                if l < L - 2:
                    B = np.dot(perms[l + 1], theta_B[l + 1].T)
                else:
                    B = theta_B[l].T

                # Solve LAP to update P_l using the cost matrix from A and B
                # C = A + B.T
                # perms[l] = solve_lap(C, np.eye(C.shape[0])) # Update permutation matrix using both previous and next matrices
                perms[l] = solve_lap(A, B)
                print(perms[l])
            # Check for convergence by comparing to previous permutations
            if all(np.array_equal(perms[l], prev_perms[l]) for l in range(L - 1)):
                break

        return perms

    permutation_matrices = permutation_coordinate_descent(target_model, model, model_arch_info, model_config)
    print(permutation_matrices)

    writer = TensorWriter(
        out_path,
        max_shard_size=merge_options.out_shard_size,
        safe_serialization=merge_options.safe_serialization,
    )

    # Write the aligned model weights to the output path with `writer`
    for layer_index, weight_info in enumerate(model_arch_info.all_weights(model_config)):
        if layer_index < len(permutation_matrices):
            # Retrieve the original weight tensor from model B
            original_tensor = _get_tensor(model, weight_info.name)

            # Apply the permutation matrix to the weight tensor
            permuted_tensor = torch.matmul(permutation_matrices[layer_index], original_tensor)

            # Save the permuted tensor using the TensorWriter
            writer.save_tensor(weight_info.name, permuted_tensor, clone=True)
        else:
            # For layers without a permutation matrix, just save the original tensor
            tensor = _get_tensor(model, weight_info.name)
            writer.save_tensor(weight_info.name, tensor, clone=True)

    writer.flush_current_shard()  # Flush the last shard to ensure all tensors are written
    writer.close()


if __name__ == "__main__":
    with torch.no_grad():
        main()