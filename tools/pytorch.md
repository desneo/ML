# [1.Pytorch doc](https://pytorch.org/docs/stable/index.html)
## [1.1 torch.nn](https://pytorch.org/docs/stable/nn.html)
### [nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)
```
Parameters:
   in_channels (int) – Number of channels in the input image
   out_channels (int) – Number of channels produced by the convolution
   kernel_size (int or tuple) – Size of the convolving kernel
   stride (int or tuple, optional) – Stride of the convolution. Default: 1
   padding (int or tuple, optional) – Zero-padding added to both sides of the input. Default: 0
   padding_mode (string, optional) – 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
   dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
   groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
   bias (bool, optional) – If True, adds a learnable bias to the output. Default: True
```` 
### [nn.MaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d)
```
Parameters:
    kernel_size – the size of the window to take a max over
    stride – the stride of the window. Default value is kernel_size
    padding – implicit zero padding to be added on both sides
    dilation – a parameter that controls the stride of elements in the window
    return_indices – if True, will return the max indices along with the outputs. Useful for torch.nn.MaxUnpool2d later
    ceil_mode – when True, will use ceil instead of floor to compute the output shape
```
