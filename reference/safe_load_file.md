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
#> -0.7678  0.7275  1.1920 -1.2495  0.4722  1.6192  0.2329  0.1686  2.2717 -2.0607
#> -0.9029  0.1503 -0.7809 -1.0870  1.2114  0.9040 -1.7078 -1.4612  0.6955 -0.5991
#> -1.1957 -0.8127  1.1384 -0.0299 -0.5566  0.8371  0.4054  0.6655 -0.0448  0.4784
#>  0.5880 -0.0730 -1.1767  0.2360 -0.5156  0.0310 -0.9280 -0.2278 -0.0502 -0.3248
#> -0.4118  0.6974 -0.8161 -0.7003 -1.6859 -0.0306 -0.1337  1.2460  0.3290  0.0979
#> -0.1638 -0.8297  0.5718 -1.6156  0.5921  1.9441 -0.6383 -0.3186  1.2504 -0.2567
#>  0.8781 -0.6482  0.2723 -1.8116  0.5301  1.3791 -0.1621  0.6621 -1.3489 -0.7308
#> -0.7855 -0.2634  1.0209 -0.6950  0.3652  0.4107  1.7878 -0.1367 -0.4852 -0.2258
#>  1.3683  0.1308 -0.2051 -0.3827 -1.8180 -0.9688 -0.3143 -0.0627  0.7242 -1.4031
#> -0.9546  0.9397 -1.1547 -0.5306  0.3414  0.1527 -0.4534 -1.3598  0.6195  0.2630
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
