# Safe load a safetensors file

Loads an safetensors file from disk.

## Usage

``` r
safe_load_file(path, ..., framework)
```

## Arguments

- path:

  Path to the file to load

- ...:

  Additional framework dependent arguments to pass to the tensor
  creation function.

- framework:

  Framework to load the data into. Currently supports "torch" and "pjrt"

## Value

A list with tensors in the file. The `metadata` attribute can be used to
find metadata the metadata header in the file.

## See also

[safetensors](https://mlverse.github.io/safetensors/reference/safetensors.md),
[`safe_save_file()`](https://mlverse.github.io/safetensors/reference/safe_save_file.md)

## Examples

``` r
if (rlang::is_installed("torch") && torch::torch_is_installed()) {
  tensors <- list(x = torch::torch_randn(10, 10))
  temp <- tempfile()
  safe_save_file(tensors, temp)
  safe_load_file(temp, framework = "torch")
}
#> $x
#> torch_tensor
#>  0.1182 -0.8531  0.1907  0.1769  0.9274  2.1357 -0.8121  0.2269  0.2435 -0.1391
#> -2.0917 -0.9227 -0.5378  0.5014  1.0973  1.1200 -1.9894 -0.7039  0.4397  0.0358
#>  0.0802  0.2162  0.7101  0.0438 -0.3039  0.4807  1.3861 -1.7354  2.2889  0.0338
#> -0.5343  1.0673  0.4292 -1.1841  1.6993  2.3233  1.2905 -0.2680 -1.5197  0.3339
#>  0.0808  0.0153  0.1027  0.7539 -1.5537 -0.7829 -0.1562 -0.3894  0.1717 -0.2484
#>  0.1285 -0.8801  0.6706  1.5681 -0.0967  0.0392 -0.5867 -1.4180 -0.5370  0.7124
#> -0.9460 -1.1076  0.9546  2.6231 -1.1928  0.7284  2.3517 -0.2620 -0.0615  1.2114
#>  0.4848 -0.7939 -1.8170  0.9171  1.1498 -0.8194  0.1718  1.1660  0.6852 -0.3900
#>  0.0645  0.1774 -0.9329 -0.4713 -1.1073  0.5666 -0.9292 -0.6196  0.5366 -0.6839
#>  1.1481 -2.5841  0.4889 -0.7939  1.1609 -0.3914  0.2867 -1.7737 -0.6064 -0.0553
#> [ CPUFloatType{10,10} ]
#> 
#> attr(,"metadata")
#> attr(,"metadata")$x
#> attr(,"metadata")$x$shape
#> [1] 10 10
#> 
#> attr(,"metadata")$x$dtype
#> [1] "F32"
#> 
#> attr(,"metadata")$x$data_offsets
#> [1]   0 400
#> 
#> 
#> attr(,"max_offset")
#> [1] 468
```
