# FAQ
```
读写二进制文件
  a = np.fromfile('out_alpha_0.bin', dtype=np.float32)
  np.savetxt("a.new", np.reshape(a,(-1,16)), delimiter = ' ', fmt = '%.5f')

图片转np
  img = np.asarray(Image.open(self.files[idx]))
```
