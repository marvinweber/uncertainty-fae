import torch


class TensorList(list):
    """A List of Tensors implementing basic Tensor operations (Wrapper to use as Model Input)
    
    TODO Docs
    """

    def to(self, *args, **kwargs) -> 'TensorList':
        moved_tensors = []
        for tensor in self:
            if isinstance(tensor, torch.Tensor):
                tensor_moved = tensor.to(*args, **kwargs)
            moved_tensors.append(tensor_moved)

        return TensorList(moved_tensors)

    def cuda(self, *args, **kwargs) -> 'TensorList':
        moved_tensors = []
        for tensor in self:
            if isinstance(tensor, torch.Tensor):
                tensor_moved = tensor.cuda(*args, **kwargs)
            moved_tensors.append(tensor_moved)

        return TensorList(moved_tensors)

    def cpu(self, *args, **kwargs) -> 'TensorList':
        moved_tensors = []
        for tensor in self:
            if isinstance(tensor, torch.Tensor):
                tensor_moved = tensor.cpu(*args, **kwargs)
            moved_tensors.append(tensor_moved)

        return TensorList(moved_tensors)

    @property
    def device(self):
        if len(self) == 0:
            raise ValueError('Empty TensorList cannot be located on device!')
        dev = self[0].device
        if any(dev != t.device for t in self):
            raise ValueError('Tensors of TensorList are on different devices!')
        return dev
