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
#>  1.5366 -0.4293  0.1923  0.8389  0.5581 -0.1003 -1.2019  1.1080  0.1084  0.7487
#> -0.0435 -0.0204  0.0575 -0.2682 -1.3316  0.5817 -0.4627 -1.0268  0.7121 -0.5215
#>  1.2859  1.1296  0.0017 -0.2271  1.1849  0.0105 -0.3420  0.4540  0.4503 -0.1899
#>  0.0139  0.0917  0.3333  0.8178  0.3728  0.1421 -0.6978 -1.5009 -1.0247 -0.1409
#> -0.2503 -0.3031  0.6320 -0.9945 -0.6237 -2.0303 -1.6441  0.7358  0.9725  0.6453
#>  0.1470  0.6450 -0.3576 -0.8405 -0.7844  0.3810  1.3380 -1.6382 -0.5668 -1.2576
#>  0.1130  0.2480  1.3269 -1.3109  0.6230  0.4487  2.1071  0.6574  2.0793 -0.3555
#>  1.3174  0.4284 -1.1626 -1.1710  0.4863  0.2197 -1.2037 -0.1533  0.8557 -2.2929
#>  0.5238 -0.2375  0.3469 -1.0322 -1.2331  0.1668 -1.3805 -0.3497 -0.9861  0.9863
#>  0.4631  0.0695 -1.0960 -2.3412 -0.1432  0.2758 -0.3004 -0.2042 -0.6671 -0.8315
#> [ CPUFloatType{10,10} ]
```
