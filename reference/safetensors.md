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
#> -0.7607 -1.5903  1.3032  0.9316 -1.4465 -0.0276  0.8147  0.3435  0.2893  0.3427
#> -0.2504  0.4017 -0.4147 -0.6876 -0.6048 -1.8019 -1.3750 -0.3459 -0.0811 -0.6737
#> -0.3351  0.6990  0.5775 -0.7202  1.1339  0.6388 -0.6640  0.9807 -0.8039 -0.1183
#> -0.0491 -0.7015 -0.4361  1.5536 -1.6699 -0.8996 -1.1626  1.1475  0.3878 -0.3586
#> -0.9219  0.5434  0.0741 -0.0662 -1.7796  0.9304  0.0462 -0.1227 -0.1066  0.7292
#> -0.7618  0.7720  0.2380 -0.4670 -1.6823  0.5310 -0.3117 -0.1711  0.2250  0.6729
#> -0.4805 -1.3447 -1.9685  0.5914  0.6671  1.0179  1.0499  0.1045 -0.7571  0.4076
#>  0.2441 -0.9262  0.3614 -0.9845 -0.9956  0.2952 -1.8597  1.3244 -0.4285  0.4008
#>  2.0253 -0.2042  1.5491 -0.9338  1.3587  1.1942  1.5286 -0.5165 -0.4576 -0.8911
#>  1.1031 -1.5025 -0.6454 -1.9580 -0.0438  0.6038  0.7836 -1.5237 -0.2082  0.3419
#> [ CPUFloatType{10,10} ]
```
