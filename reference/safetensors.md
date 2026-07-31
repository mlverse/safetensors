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
#> -1.0627 -1.2559 -0.9041  0.4557  0.1686 -0.6656 -1.3845  1.7669 -0.4100 -0.3130
#> -0.1271  1.9296 -0.5429  0.7826 -2.9803  0.8287 -0.9888 -1.4505 -0.5090 -1.9625
#>  0.4679 -0.5841  0.0061  0.3742 -2.3503 -0.2907  0.0645 -0.5015  0.1224 -0.8582
#> -0.2292  0.8113 -1.5984 -1.0699 -0.4390  0.6313 -0.2561  0.0766  1.7489  0.8321
#> -1.3691  1.0099 -0.9749 -0.9057 -0.8797  0.3959  0.9951 -0.3783 -1.2241 -0.7235
#>  0.4438  0.4611  0.4105  1.0518  0.6537 -0.7473  0.4545  0.4615 -1.6793 -1.5992
#>  0.3768 -0.5308  0.3865  0.0690 -0.5512  1.2574 -1.2083  0.4370 -2.2559 -0.5618
#>  1.1449  1.9546  1.4247 -0.7802  1.1506  0.4656 -0.1431 -0.5527  0.9329 -0.5327
#> -0.4201  0.5140 -0.7615 -0.1490 -1.2462  1.0648  0.1058 -0.6202 -1.6793 -0.6376
#>  0.4679  0.9488 -0.3395  2.1389 -0.2068 -0.8514 -0.8674  0.3893 -0.3418 -1.0490
#> [ CPUFloatType{10,10} ]
```
