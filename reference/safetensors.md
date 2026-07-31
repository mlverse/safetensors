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
#> -0.6860 -0.9856 -1.2685 -0.6387 -0.5339  1.3239  0.3230  0.1494  1.6089  0.3227
#> -0.9800  0.6593  1.1458 -0.8423 -0.3907  1.2718 -0.4677 -0.6466  1.5220 -0.9672
#> -0.0587 -1.3510  1.5987 -2.0557 -0.3404 -0.1433 -1.1478  0.6112  1.0442 -0.1269
#> -0.1373 -0.6863 -0.0257  1.3561 -1.3183  1.9602  1.1251  0.6749  0.1621  1.9116
#>  0.9997  1.9285 -0.6922 -0.1409 -0.7547 -0.3205  0.0395  0.3903 -0.0676  0.7908
#>  0.4628 -0.0186 -1.6940  0.8960  0.2475 -1.2415 -0.8001  1.7488 -1.1562  0.7671
#>  0.2656 -1.3974 -0.3967  0.4401  0.6045 -0.5047 -0.7223 -0.5993  0.3624  0.1249
#> -0.8090 -1.1976 -0.4965 -0.4363 -0.3370 -0.4202  0.8323 -2.3746  0.9691 -1.9257
#> -0.8503  1.5954 -0.5697  0.4015 -1.7703 -1.6393  0.6484  0.0717  0.1191  0.4053
#>  0.9575 -1.0187  0.4803 -2.0411 -1.2909 -0.0156 -0.4849 -0.8252  0.1497 -2.5427
#> [ CPUFloatType{10,10} ]
```
