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
#> -1.9812 -0.1393  0.3857  2.2306  0.4043  0.7727 -2.2690  1.1313  0.0648  0.3550
#> -0.2983 -0.2911  0.5260  1.5157 -0.2049 -0.0608 -2.0061 -1.2914 -0.9953  0.0199
#> -0.6847 -0.8500  0.2571 -1.5001  1.1277  0.4124 -0.6230 -0.9497 -0.0542  0.2955
#> -0.6891 -2.5649  0.6974 -0.1576 -0.0903 -0.6208 -0.1606 -0.4050  0.5975 -2.0281
#>  0.1543 -1.3898 -2.7762 -0.9342 -0.1838  0.6150 -0.9096  0.5856 -0.6742 -0.2279
#>  0.6509 -1.2123  0.1762  0.1677 -0.8946 -0.0364 -0.6294  0.5315 -0.1137 -1.3235
#>  0.3490 -0.4840 -1.9062 -0.2959 -0.0437 -0.0963  0.8296 -1.2014 -1.0627  1.8291
#>  0.6675 -2.8990  1.1669 -1.3053  0.1601  0.1165  0.1531 -3.0289  0.4855  0.6451
#> -0.9689  0.5073  0.2545  0.9410 -0.0577  0.5797  2.0804 -0.3417 -0.0394  1.3199
#> -0.1867  0.9182  0.2628  0.2934 -0.4498 -1.0771  1.1775 -1.5232  1.2347 -0.0811
#> [ CPUFloatType{10,10} ]
```
