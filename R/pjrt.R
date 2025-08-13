pjrt_dtype_to_safetensors <- function(pjrt_dtype) {
  switch(
    as.character(pjrt_dtype),
    "pred" = "BOOL",
    "i8" = "I8",
    "i16" = "I16",
    "i32" = "I32",
    "i64" = "I64",
    "ui8" = "U8",
    "ui16" = "U16",
    "ui32" = "U32",
    "ui64" = "U64",
    "f32" = "F32",
    "f64" = "F64",
    cli::cli_abort("Unsupported PJRT data type {.val {pjrt_dtype}}")
  )
}

safetensors_dtype_to_pjrt <- function(safetensors_dtype) {
  switch(
    safetensors_dtype,
    "BOOL" = "pred",
    "I8" = "i8",
    "I16" = "i16",
    "I32" = "i32",
    "I64" = "i64",
    "U8" = "ui8",
    "U16" = "ui16",
    "U32" = "ui32",
    "U64" = "ui64",
    "F32" = "f32",
    "F64" = "f64",
    cli::cli_abort(
      "Unsupported safetensors data type {.val {safetensors_dtype}}"
    )
  )
}

pjrt_dtype_size <- function(pjrt_dtype) {
  switch(
    pjrt_dtype,
    "pred" = 1L,
    "i8" = 1L,
    "i16" = 2L,
    "i32" = 4L,
    "i64" = 8L,
    "ui8" = 1L,
    "ui16" = 2L,
    "ui32" = 4L,
    "ui64" = 8L,
    "f32" = 4L,
    "f64" = 8L,
    "pred" = 1L,
    cli::cli_abort("Unsupported PJRT data type {.val {pjrt_dtype}}")
  )
}

#' @export
safe_tensor_buffer.PJRTBuffer <- function(x) {
  pjrt::as_raw(x, row_major = TRUE)
}

#' @export
safe_tensor_meta.PJRTBuffer <- function(x) {
  list(
    shape = as.list(dim(x)), # Convert to list to avoid simplification
    dtype = pjrt_dtype_to_safetensors(as.character(pjrt::element_type(x)))
  )
}
