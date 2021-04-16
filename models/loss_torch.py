import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, L1Loss


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def compute_per_channel_dice(input, target, epsilon=1e-5,
                             ignore_index=None, weight=None):
    # assumes that input is a normalized probability
    # input and target shapes must match
    #assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # mask ignore_index if present
    if ignore_index is not None:
        mask = target.clone().ne_(ignore_index)
        mask.requires_grad = False

        input = input * mask
        target = target * mask

    input = flatten(input)
    target = flatten(target)

    target = target.float()
    # Compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    denominator = (input + target).sum(-1)
    return 2. * intersect / denominator.clamp(min=epsilon)


def dice_metric(logits, labels):
    outputs = nn.Softmax(dim=1)(logits)
    labels = expand_as_one_hot(labels, C=outputs.size()[1])
    per_channel_dice = compute_per_channel_dice(outputs, labels)
    return per_channel_dice


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """
    #assert input.dim() == 4

    # expand the input tensor to Nx1xDxHxW before scattering
    #input = input.unsqueeze(1)
    # create result tensor shape (NxCxDxHxW)
    shape = 5
    #shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
    else:
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)

    return result
import tensorflow as tf

def DiceLoss(input, target):
    """Computes Dice Loss, which just 1 - DiceCoefficient described above.
    Additionally allows per-class weights to be provided.
    """
    #self.normalization = nn.Softmax(dim=1)
        # if True skip the last channel in the target
            # get probabilities from logits
    #input = self.normalization(input)
    weight = torch.Tensor([1.0, 1.0])

    if weight is not None:
        weight = Variable(weight, requires_grad=False)
    else:
        weight = None

    #target = expand_as_one_hot(torch.Tensor(target), C=input.shape[1])

    per_channel_dice = compute_per_channel_dice(torch.Tensor(input), torch.Tensor(target), epsilon=1e-5, ignore_index=None,
                                                    weight=weight)
        # Average the Dice score across all channels/classes
    return torch.mean(1. - per_channel_dice), per_channel_dice