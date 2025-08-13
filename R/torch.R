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

#' @export
safe_tensor_buffer.torch_tensor <- function(x) {
  torch::buffer_from_torch_tensor(x$cpu())
}

#' @export
safe_tensor_meta.torch_tensor <- function(x) {
  list(
    shape = as.list(x$shape), # we must store as a list to avoid simplification
    dtype = torch_dtype_to_safe(x$dtype)
  )
}

torch_tensor_from_raw <- function(raw, meta, device = "cpu") {
  x <- torch::torch_tensor_from_buffer(
    raw,
    shape = meta$shape,
    dtype = torch_dtype_from_safe(meta$dtype)
  )
  if (device == "cpu") {
    x$clone() # we need to explicitly clone in case the device is cpu
  } else {
    x$to(device = device)
  }
}

torch_dtype_from_safe <- function(x) {
  switch(
    x,
    "F16" = "float16",
    "F32" = "float",
    "F64" = "float64",
    "BOOL" = "bool",
    "U8" = "uint8",
    "I8" = "int8",
    "I16" = "int16",
    "I32" = "int32",
    "I64" = "int64",
    "BF16" = "bfloat16",
    cli::cli_abort("Unsupported dtype {.val {x}}")
  )
}
