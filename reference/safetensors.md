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
#> -1.0732 -1.7309  1.2623  0.6782  0.7817  0.3364 -0.3470  1.4112 -0.9299 -0.3635
#>  1.4421  0.1589 -0.4258  0.2720  0.4895  0.4693 -0.5194 -0.0487 -1.3443 -0.0391
#> -0.2145  0.1238 -0.4772  0.0750 -0.4611  0.9282 -0.1520 -0.4626  1.0274 -1.2479
#>  1.0089  0.7766 -1.8261  2.6266 -0.1874  2.0026  0.9857  0.5338 -1.1062 -0.2821
#> -1.0357  2.0158  0.4160 -0.3852  1.4835  0.0751  0.3246 -0.6831  2.6949 -0.3892
#>  1.6396 -0.2560 -2.0943  3.0449  0.2376  0.3760  1.0628 -0.0572 -0.2745  2.1988
#>  1.1972  0.2064 -1.1945  2.3122 -0.4933 -0.7780  0.1879 -1.0606  0.8658 -0.6540
#>  0.9759  0.0265  0.3974  1.0740 -1.0321  1.0843  0.5731 -0.2314  1.3697 -1.2858
#>  1.3207  0.9408  0.2447  0.8876 -0.8155  0.8217 -0.4064  1.4307  0.7048  0.3852
#> -1.8168  0.7169 -0.0027  1.4499  1.1732  0.4406 -0.5654 -0.0825  0.7695 -0.0879
#> [ CPUFloatType{10,10} ]
```
