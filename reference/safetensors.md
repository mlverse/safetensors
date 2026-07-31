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
#>  0.6470 -0.1778 -1.2655  0.6781  0.0727  0.5978  0.3531  1.1472  1.4527  0.7611
#>  0.4270 -0.4205  0.8801 -0.3817  0.3640  0.9190 -0.6502  1.7410  0.5177 -1.2479
#> -0.2878 -1.1461  0.2236  1.2577  0.4943 -0.8800  0.2936  0.1322 -0.1442  0.1596
#> -2.1332 -1.0950  1.6229  0.4629  0.9012  0.0980  0.9294 -0.1514 -0.5579 -2.3730
#>  0.0945 -0.9646  1.3968 -0.1990 -0.4371  0.3497  0.1756 -0.2806  0.6681 -0.4007
#>  0.1414 -0.9226 -0.8606  1.4665  0.5474  0.2606 -1.2576 -0.6526  0.0051 -0.6335
#> -1.0689  1.7021 -1.8743 -0.9899 -0.1856 -0.6308 -1.8736 -0.5937 -0.1767 -1.3625
#> -0.8586  0.0174 -2.4019 -2.2758  0.2713 -0.1178 -0.4450 -0.7123 -0.6940  0.0593
#>  0.4176 -0.3355 -0.9010  0.2116 -0.0737  0.3448 -0.7673  0.7414 -0.8273  0.5378
#>  0.0590 -1.1824 -0.4810 -0.2369  2.0622  0.8497 -0.0107  0.2338 -1.6839  1.2790
#> [ CPUFloatType{10,10} ]
```
