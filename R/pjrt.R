#' Convert PJRT Data Type to Safetensors Format
#'
#' Maps PJRT data types to safetensors format strings.
#' Supports all PJRT data types that have safetensors equivalents.
#'
#' @param pjrt_dtype PJRT data type string (e.g., "f32")
#'
#' @return Safetensors data type string (e.g., "F32")
#' @export
pjrt_dtype_to_safetensors <- function(pjrt_dtype) {
  switch(
    pjrt_dtype,
    "pred" = "BOOL",
    "s8" = "I8",
    "s16" = "I16",
    "s32" = "I32",
    "s64" = "I64",
    "u8" = "U8",
    "u16" = "U16",
    "u32" = "U32",
    "u64" = "U64",
    "f32" = "F32",
    "f64" = "F64",
    cli::cli_abort("Unsupported PJRT data type {.val {pjrt_dtype}}")
  )
}

#' Convert Safetensors Data Type to PJRT Format
#'
#' Maps safetensors data type strings to PJRT format.
#' Supports all safetensors data types that have PJRT equivalents.
#'
#' @param safetensors_dtype Safetensors data type string (e.g., "F32")
#'
#' @return PJRT data type string (e.g., "f32")
#' @export
safetensors_dtype_to_pjrt <- function(safetensors_dtype) {
  switch(
    safetensors_dtype,
    "BOOL" = "pred",
    "I8" = "s8",
    "I16" = "s16",
    "I32" = "s32",
    "I64" = "s64",
    "U8" = "u8",
    "U16" = "u16",
    "U32" = "u32",
    "U64" = "u64",
    "F32" = "f32",
    "F64" = "f64",
    cli::cli_abort("Unsupported safetensors data type {.val {safetensors_dtype}}")
  )
}

#' Get Element Size in Bytes for PJRT Data Type
#'
#' Returns the size in bytes for a given PJRT data type.
#'
#' @param pjrt_dtype PJRT data type string
#'
#' @return Integer size in bytes
#' @export
pjrt_dtype_size <- function(pjrt_dtype) {
  switch(
    pjrt_dtype,
    "pred" = 1L,
    "s8" = 1L,
    "s16" = 2L,
    "s32" = 4L,
    "s64" = 8L,
    "u8" = 1L,
    "u16" = 2L,
    "u32" = 4L,
    "u64" = 8L,
    "f16" = 2L,
    "f32" = 4L,
    "f64" = 8L,
    "pred" = 1L,
    cli::cli_abort("Unsupported PJRT data type {.val {pjrt_dtype}}")
  )
}

tensor_buffer.PJRTBuffer <- function(x) {
  pjrt::as_raw(x, row_major = TRUE)
}

tensor_meta.PJRTBuffer <- function(x) {
  if (!inherits(x, "PJRTBuffer")) {
    cli::cli_abort("Expected a PJRT buffer object")
  }

  # Get dimensions
  dims <- dim(x)

  # Get element type and convert to safetensors format
  element_type <- pjrt::pjrt_element_type(x)
  dtype <- pjrt_dtype_to_safetensors(as.character(element_type))

  list(
    shape = as.list(dims),  # Convert to list to avoid simplification
    dtype = dtype
  )
}
