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
#> -0.4665 -1.3515 -2.3156  0.5400  1.3035 -0.2202 -0.0179  0.3061  0.1290 -1.2483
#> -1.1907 -0.0116  0.8464  0.8540  1.0643 -0.6142 -1.5320  0.3088  0.7755  0.5717
#>  0.0600 -0.2177  1.9722 -1.7648 -1.5127 -1.4779 -1.4053  1.2353  0.1851  0.2423
#> -0.6620  0.4360 -1.5430 -3.0597  0.9849  0.7074 -1.2283 -0.9638  1.7575  1.9137
#>  0.8695  0.5490 -0.8007 -0.1137 -0.5506  0.2257  0.4070 -0.4087  0.1568  0.1361
#> -1.3441  1.6087 -0.9905  0.0520  1.0383  0.3677  1.4192  2.2033  0.3175  0.7485
#> -1.3282  0.1335  0.5507  1.2367 -0.5294  0.3393 -0.7898 -1.6097 -0.7051 -0.7347
#> -1.0343  0.7644 -1.3423  0.9238  1.8487  2.3557  1.0027 -1.7798 -1.0271  2.1576
#> -0.6067  1.4470  1.7061 -0.9366  1.0468  0.1873  0.2071  1.2047  0.5803  0.0626
#> -0.6521 -0.9365  0.8829  2.1827  0.0096 -0.1451  0.1733  1.3962 -0.5977 -0.2898
#> [ CPUFloatType{10,10} ]
```
