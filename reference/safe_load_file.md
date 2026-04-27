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
#>  0.3466  0.3958  0.4040  1.1250  0.3048 -0.9167 -0.2940 -0.0906 -0.6828  1.1636
#> -0.7535  0.6077 -0.7574  1.3965  0.4393 -1.4191  1.4202  0.4830  0.4408 -0.5577
#>  1.3301 -0.2350 -0.0213  0.2041 -0.4305 -2.4779  1.2951  0.3858  0.3291  0.2862
#>  0.4002  1.1302  0.1412 -0.7997 -0.1912 -0.2885  0.3793  0.8263 -1.4362  1.3615
#> -1.0459 -1.4348 -1.0202 -0.1688 -0.6395  0.7309  0.7742  0.0705  0.1985 -0.2709
#> -0.2159 -0.0492  1.8888  1.4265 -0.8724  0.1873  0.3559  2.1889  0.7390  0.0788
#>  0.6875 -1.3202  0.8391 -0.2170 -0.5714  0.6659 -0.9905  0.3067 -0.7189 -1.1898
#>  0.2238  0.2021 -0.4330  0.4661  0.9189  0.4211  0.2726  1.1223 -0.5253  0.3511
#>  0.7359  1.4722 -1.0034 -0.7675  0.9104 -1.1443 -0.4213  0.2169  0.2854  0.7875
#> -0.3726  0.0195 -0.4832  2.7928  0.2716  0.7693 -0.8475  1.1150 -0.7152 -1.3187
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
