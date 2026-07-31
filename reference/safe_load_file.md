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
#>  0.8656  0.3070  0.4011 -0.8461 -0.8740 -0.2651  0.8225  1.4995 -1.7418  0.7966
#> -0.0724 -0.0686  0.6639  0.7386 -0.5095 -1.2718 -1.0354  1.4428  1.3319 -1.0646
#>  1.2936  0.2153  0.3361  0.2047  0.0116  0.3176  0.1903  0.6833  0.5078  2.3843
#> -0.8062 -1.1040  0.5867  0.9829 -0.0004 -0.1923 -1.6419 -0.0652 -1.3904  0.0121
#> -0.3232  0.6791  0.2190 -0.7582  0.6832 -1.1353  0.9972  0.7064 -0.2580 -1.0708
#> -0.5580  0.1995 -0.2761  2.7544 -2.4598  1.7510  1.6460  0.0085 -0.0455  1.0123
#>  1.0631  0.2365 -0.8378 -0.0914 -1.1895  0.6359 -0.1791 -0.4714  0.5784 -0.6197
#> -2.3765  0.6864  1.1580 -0.2470  0.5557  1.9871  1.2598  0.9874  1.6027 -0.3071
#> -1.1136  0.8556  1.3991  0.5402 -0.3202 -1.0698  0.7172 -1.6517  1.3995 -0.5169
#>  1.8182 -2.0727  0.2951  0.7390 -1.6156  0.3589 -0.3440 -0.2617  0.4863  1.1237
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
