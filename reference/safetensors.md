# Low level control over safetensors files

Low level control over safetensors files

Low level control over safetensors files

## Details

Allows opening a connection to a safetensors file and query the tensor
names, metadata, etc. Opening a connection only reads the file metadata
into memory. This allows for more fined grained control over reading.

## Public fields

- `con`:

  the connection object with the file

- `metadata`:

  an R list containing the metadata header in the file

- `framework`:

  the framework used to return the tensors

- `args`:

  additional arguments for tensor creation

- `max_offset`:

  the largest offset boundary that was visited. Mainly used in torch to
  find the end of the safetensors file.

## Methods

### Public methods

- [`safetensors$new()`](#method-safetensors-new)

- [`safetensors$keys()`](#method-safetensors-keys)

- [`safetensors$get_tensor()`](#method-safetensors-get_tensor)

- [`safetensors$clone()`](#method-safetensors-clone)

------------------------------------------------------------------------

### Method `new()`

Opens the connection with the file

#### Usage

    safetensors$new(path, ..., framework)

#### Arguments

- `path`:

  Path to the file to load

- `...`:

  (any)  
  Additional, framework dependent, arguments to pass to use when
  creating the tensor. For torch, this is the device, for pjrt the
  client.

- `framework`:

  Framework to load the data into. Currently supports "torch" and "pjrt"

------------------------------------------------------------------------

### Method `keys()`

Get the keys (tensor names) in the file

#### Usage

    safetensors$keys()

------------------------------------------------------------------------

### Method `get_tensor()`

Get a tensor from its name

#### Usage

    safetensors$get_tensor(name)

#### Arguments

- `name`:

  Name of the tensor to load

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    safetensors$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
if (rlang::is_installed("torch") && torch::torch_is_installed()) {
tensors <- list(x = torch::torch_randn(10, 10))
temp <- tempfile()
safe_save_file(tensors, temp)
f <- safetensors$new(temp, framework = "torch")
f$get_tensor("x")
}
#> torch_tensor
#>  0.3188 -0.0070 -1.1983  0.9794  0.0976  0.6959 -1.1052  0.5176  0.9365  1.0373
#> -0.1416 -1.3537  0.5852 -1.3820  1.1696 -0.5394 -0.9699  0.3192 -0.1863  0.6261
#>  1.3077 -0.4161  1.1374  0.0031 -1.0907 -0.4959 -1.3763  1.8197  1.3162  0.6872
#> -1.2117  0.9898  0.9923  0.0100 -0.4432 -1.6014 -0.2431 -0.6390  0.4188  1.0861
#> -0.8629  1.5807 -0.4309  0.8042 -0.0405 -0.1282 -1.4884  0.1367  0.9447  0.1024
#> -0.0134 -0.0430  0.6434 -0.2506  0.5348 -0.8123  0.7483  0.3872 -0.7514 -0.9095
#>  0.9356 -0.8489 -0.3783 -1.0463  0.3368 -0.3203  0.6125 -0.1302  2.4414 -0.9573
#> -1.8860  0.8373 -1.6818 -2.6007 -0.5773  0.9267  0.8566 -0.8566  0.2864 -1.2279
#> -1.3648  0.0552  0.4790 -0.4219  0.3860 -0.9099 -0.0599  0.4664  0.5828  0.8926
#>  2.3897  0.5572 -0.6910  0.8843  1.0823 -0.3760  1.0118  0.6096 -0.0894 -1.1114
#> [ CPUFloatType{10,10} ]
```
