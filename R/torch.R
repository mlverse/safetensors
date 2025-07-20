torch_dtype_to_safe <- function(x) {
  if (x == torch::torch_float()) {
    return("F32")
  } else if (x == torch::torch_float16()) {
    return("F16")
  } else if (x == torch::torch_float64()) {
    return("F64")
  } else if (x == torch::torch_bool()) {
    return("BOOL")
  } else if (x == torch::torch_uint8()) {
    return("U8")
  } else if (x == torch::torch_int8()) {
    return("I8")
  } else if (x == torch::torch_int16()) {
    return("I16")
  } else if (x == torch::torch_int32()) {
    return("I32")
  } else if (x == torch::torch_int64()) {
    return("I64")
  } else if (x == torch::torch_bfloat16()) {
    return("BF16")
  } else if (x == torch::torch_cfloat()) {
    return("C64")
  } else if (x == torch::torch_cdouble()) {
    return("C128")
  } else {
    cli::cli_abort("Unsupported data type {.val {x}}")
  }
}

tensor_buffer.torch_tensor <- function(x) {
  torch::buffer_from_torch_tensor(x$cpu())
}

tensor_meta.torch_tensor <- function(x) {
  list(
    shape = as.list(x$shape), # we must store as a list to avoid simplification
    dtype = torch_dtype_to_safe(x$dtype)
  )
}
