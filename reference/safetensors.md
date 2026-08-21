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
#> -0.0859 -0.4699 -0.3618  0.4879  1.2007  0.6253  1.5825  0.3662  1.9018 -0.5065
#>  1.1967  0.6674 -0.8635  0.4227 -0.5435 -0.2583 -0.4041  0.3357  0.7595 -0.3544
#> -0.1810  0.1976 -1.5687  1.0012  0.8850  0.9547 -1.1262 -1.3265 -1.1989 -0.8853
#>  0.1182  0.4915  0.4750 -2.2251 -0.6519 -0.7920  0.0108  0.8510 -0.0425  0.1719
#>  0.4700 -0.4383  1.1404  0.4189  1.1641  1.6880 -1.2937  1.0130  0.5164 -1.7353
#> -0.6424 -0.7253 -1.4738  0.2226  0.1945 -1.7639  0.6428  1.3007  1.0251  0.6145
#> -1.1829 -0.1652  0.0710  0.0434 -0.8188 -1.1743  0.0091 -0.3920 -0.9912 -0.2551
#> -0.9098 -1.2301  0.0554 -0.0470 -0.2966 -1.3951 -0.0121 -1.4945  0.5068  1.1256
#> -0.9310  1.8581 -0.5695  1.1628  0.7361 -0.9170  0.4180 -2.0439  1.0444 -0.9508
#> -0.9715  0.6572 -0.1390  0.8256 -0.6936  0.7938  0.3901 -0.3724  0.1759 -0.2480
#> [ CPUFloatType{10,10} ]
```
