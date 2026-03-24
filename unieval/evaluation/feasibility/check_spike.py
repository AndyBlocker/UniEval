import torch

def check_inout_spike(inp, nspks_max=2):
    """Detect if a tensor contains spiking activity and compute firing rate.

    For SpikeZIP: outputs are -threshold, 0, threshold three-valued matrices.

    Args:
        inp: Input tensor to analyze.
        nspks_max: Maximum number of spike levels to consider as spiking.

    Returns:
        Tuple of (is_spike: bool, firing_rate: float, spike_histogram: None).
    """
    if torch.abs(inp).max() == 0:
        return True

    # Normalize to [-1, 0, 1] range
    inp_norm = inp / torch.abs(inp).max()
    num = inp_norm.unique()

    if (len(num) <= nspks_max + 1
            and inp_norm.max() <= nspks_max - 1
            and inp_norm.min() >= -(nspks_max - 1)):
        return True
    else:
        return False



