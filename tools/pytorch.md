# [1.Pytorch doc](https://pytorch.org/docs/stable/index.html)
## FAQ
```
    label_tendor = torch.from_numpy(label)    // ndArray转tensor
```



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

### [torch.nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear)
```
Parameters:
    in_features – size of each input sample
    out_features – size of each output sample
    bias – If set to False, the layer will not learn an additive bias. Default: True
```


## [torch.utils.data - 数据源](https://pytorch.org/docs/stable/data.html)
```
实际使用的示例
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

trainPath = "G:\\practice\\pycharm\\datasets\\mnistasjpg\\trainingSet\\trainingSet\\*\\*.jpg"
trainTestPath = "G:\\practice\\pycharm\\datasets\\mnistasjpg\\trainingSample\\trainingSample\\*\\*.jpg"
# 定义数据源
class MnistDataset(Dataset):
    def __init__(self, root_dir):
        self.files = glob.glob(root_dir)
        self.x = 123

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.asarray(Image.open(self.files[idx])).reshape((-1, 28, 28))
        label = os.path.abspath(self.files[idx]).split("\\")[-2]
        return img, label
```

## [torch.Tensor](https://pytorch.org/docs/stable/tensors.html)
```
tensor.shape  //输出维度
tensor.item() //输出tensor内的值
```
### tensor.view() 维度转换
```
>>> x = torch.randn(4, 4)
>>> x.size()
torch.Size([4, 4])
>>> y = x.view(16)
>>> y.size()
torch.Size([16])
>>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
>>> z.size()
torch.Size([2, 8])
```
