# coding=utf-8
"""Python module for handy functions"""
import configparser
import torch


def load_parameters(parameters_filepath):
    """
    load parameters from an *.ini file
    :param parameters_filepath: filename (absolute path)
    :return: nested dictionary of parameters
    """
    conf_parameters = configparser.ConfigParser()
    conf_parameters.read(parameters_filepath, encoding="UTF-8")
    # nested_parameters = utils.convert_configparser_to_dictionary(conf_parameters)
    nested_parameters = {s: dict(conf_parameters.items(s)) for s in conf_parameters.sections()}
    return nested_parameters


def stack_tensors(*args):
    """
     Stack arbitrary number of tensors along the first dimension
    :param args: list of tensors
    :return: tensor stacking all the input tensors
    """
    return torch.cat(args, dim=0)


def or_float_tensors(x_1, x_2):
    """
    ORs two float tensors by converting them to byte and back
    Note that byte() takes the first 8 bit after the decimal point of the float
    e.g., 0.0 ==> 0
          0.1 ==> 0
          1.1 ==> 1
        255.1 ==> 255
        256.1 ==> 0
    Subsequently the purpose of this function is to map 1s float tensors to 1
    and those of 0s to 0. I.e., it is meant to be used on tensors of 0s and 1s.

    :param x_1: tensor one
    :param x_2: tensor two
    :return: float tensor of 0s and 1s.
    """
    return (x_1.byte() | x_2.byte()).float()


def xor_float_tensors(x_1, x_2):
    """
    XORs two float tensors by converting them to byte and back
    Note that byte() takes the first 8 bit after the decimal point of the float
    e.g., 0.0 ==> 0
          0.1 ==> 0
          1.1 ==> 1
        255.1 ==> 255
        256.1 ==> 0
    Subsequently the purpose of this function is to map 1s float tensors to 1
    and those of 0s to 0. I.e., it is meant to be used on tensors of 0s and 1s.

    :param x_1: tensor one
    :param x_2: tensor two
    :return: float tensor of 0s and 1s.
    """
    return (x_1.byte() ^ x_2.byte()).float()


def clip_tensor(x, lb=0., ub=1.):
    """
    Clip a tensor to be within lb and ub
    :param x:
    :param lb: lower bound (scalar)
    :param ub: upper bound (scalar)
    :return: clipped version of x
    """
    return torch.clamp(x, min=lb, max=ub)


if __name__ == "__main__":
    print("a module to be imported by others, testing here")

    parameters = load_parameters("../parameters.ini")

    stacked_tensor = stack_tensors(torch.ones(5, 2), torch.zeros(5, 2))

    ored_tensor = or_float_tensors(torch.ones(5), torch.zeros(5))

    clipped_tensor = clip_tensor(2 * torch.rand(5) - 1)

    print(clipped_tensor)
