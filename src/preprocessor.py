import h5py
import numpy as np
from transformers import AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Scaling and Precision
def standardize_range(data, alpha):
    return data / alpha

# Rounding and Formatting
def round_and_format(data, precision):
    """
    Format each trajectory in the dataset to LLMTIME string format with fixed decimal precision.

    Parameters:
    - data: np.ndarray of shape (N, T, 2), where N is number of trajectories/systems and T is the number of time points.
    - precision: int, number of decimal places to keep.

    Returns:
    - traj_round: np.ndarray of shape (N, T, 2), where each trajectory is rounded to the specified precision.
    - traj_str: List of strings, where each string is a trajectory in LLMTIME format.
    """
    traj_round = np.round(data, decimals=precision)

    format_str = f"{{:.{precision}f}},{{:.{precision}f}};"
    traj_str = []
    for traj in traj_round: # traj is shape (T, 2)
        string = "".join(format_str.format(x, y) for x, y in traj)
        traj_str.append(string)

    return traj_round, traj_str

# Tokenization
def tokenize(string_list, tokenizer):
    """
    Convert each LLMTIME-formatted string into token ID sequence.

    Parameters:
    - string_list: list of LLMTIME-formatted strings
    - tokenizer: HuggingFace tokenizer (e.g., Qwen tokenizer)

    Returns:
    - list of token ID sequences (list of ints)
    """
    return [tokenizer(system, return_tensors="pt")["input_ids"].tolist()[0] for system in string_list]

# Load and Preprocess Split

def load_and_preprocess(h5_path: str, alpha: float = 1, precision: int = 3, train_ratio: float = 0.8, seed: int = 42):
    """
    Load and preprocess the Lotka-Volterra dataset.

    This function loads the dataset, splits it into training and validation sets, scales the data, 
    and formats it into LLMTIME string format for further processing.

    Parameters:
    - h5_path (str): Path to the HDF5 file containing the dataset.
    - alpha (float, optional): Scaling factor to normalize the data. Default is 1.
    - precision (int, optional): Number of decimal places to retain in the formatted strings. Default is 3.
    - train_ratio (float, optional): Proportion of the dataset to use for training. Default is 0.8.
    - seed (int, optional): Random seed for reproducibility of the train-validation split. Default is 42.

    Returns:
    - tuple: 
        - (train_set, train_idx): 
            - train_set (list of str): Formatted training data as LLMTIME strings.
            - train_idx (np.ndarray): Indices of the training samples in the original dataset.
        - (val_set, val_idx): 
            - val_set (tuple): Formatted validation data as a tuple of (full_string, prompt_string, target_string).
            - val_idx (np.ndarray): Indices of the validation samples in the original dataset.
    """
    # Load the dataset
    with h5py.File(h5_path, "r") as f:
        trajectories = f["trajectories"][:]  # shape (N, 100, 2)
        time_points = f["time"][:] 

    N = trajectories.shape[0]
    np.random.seed(seed)
    indices = np.random.permutation(N)
    split_idx = int(N * train_ratio)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    # Scaling
    scaled_traj = standardize_range(trajectories, alpha)

    def format_split(data_scaled):
        """
        data_scaled: np.ndarray of shape (M, 100, 2)
        Returns:
        - full_string: list of full (100-point) LLMTIME strings
        - prompt_string: list of prompt (70-point) LLMTIME strings
        - target_string: list of target (30-point) LLMTIME strings
        """
        _, full_string = round_and_format(data_scaled, precision)
        _, prompt_string = round_and_format(data_scaled[:, :70, :], precision)
        _, target_string = round_and_format(data_scaled[:, 70:, :], precision)
        return full_string, prompt_string, target_string

    val_set = format_split(scaled_traj[val_idx])
    train_set = round_and_format(scaled_traj[train_idx], precision)[1] 

    return (train_set, train_idx), (val_set, val_idx)

# Decoding and Inference

def decode_prediction_to_array(token_ids):
    """
    Decode model output tokens into a numeric (T, 2) float32 array.

    Parameters:
    - token_ids: List[int] - the output token sequence

    Returns:
    - np.ndarray of shape (T, 2)
    """
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    decoded = decoded.replace(" ", "")  # clean whitespace if needed

    points = decoded.strip().split(";")
    coords = []
    for point in points:
        if point.count(",") != 1:
            continue  # skip malformed or incomplete points
        try:
            x_str, y_str = point.split(",")
            x = float(x_str)
            y = float(y_str)
            coords.append((x, y))
        except ValueError:
            continue  # handle float conversion error

    return np.array(coords, dtype=np.float32)

def rescale(data, alpha):
    """
    Rescale the given data by multiplying it with a scaling factor alpha.

    Parameters:
    - data (np.ndarray): The input data to be rescaled。
    - alpha (float): The scaling factor to apply to the data.

    Returns:
    - np.ndarray: The rescaled data.
    """
    return data * alpha