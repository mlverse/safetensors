#' Writes a list of tensors to the safetensors format
#'
#' @param tensors A named list of tensors. Currently only torch tensors are supported.
#' @param path The path to save the tensors to. It can also be a binary connection, as eg,
#'   created with `file()`.
#' @param ... Currently unused.
#' @param metadata An optional string that is added to the file header. Possibly
#'   adding additional description to the weights.
#'
#' @examples
#' if (rlang::is_installed("torch") && torch::torch_is_installed()) {
#'   tensors <- list(x = torch::torch_randn(10, 10))
#'   temp <- tempfile()
#'   safe_save_file(tensors, temp)
#'   safe_load_file(temp)
#'
#'   ser <- safe_serialize(tensors)
#' }
#'
#' @returns The path invisibly or a raw vector.
#'
#' @export
safe_save_file <- function(tensors, path, ..., metadata = NULL) {
  if (any(duplicated(names(tensors)))) {
    cli::cli_abort("Duplicated names are not allowed in {.arg tensors}")
  }

  if (is.character(path)) {
    con <- file(path, open = "wb")
    on.exit({close(con)}, add = TRUE)
  } else {
    con <- path
  }

  write_safe(tensors, metadata, con)
  invisible(path)
}

#' @describeIn safe_save_file Serializes the tensors and returns a raw vector.
#' @export
safe_serialize <- function(tensors, ..., metadata = NULL) {
  con <- rawConnection(raw(), open = "wb")
  on.exit({close(con)}, add = TRUE)
  safe_save_file(tensors, con, metadata = metadata)
  rawConnectionValue(con)
}

write_safe <- function(tensors, metadata, con) {
  meta <- make_meta(tensors, metadata)
  meta_raw <- charToRaw(jsonlite::toJSON(meta, auto_unbox = TRUE))
  # write the metadatasize as a 64bit int

  writeBin(length(meta_raw), con = con, size = 8L)
  writeBin(meta_raw, con = con)
  for (tensor in tensors) {
    buf <- tensor_buffer(tensor)
    writeBin(buf, con = con)
  }
}

make_meta <- function(tensors, metadata) {
  meta_ <- structure(
    vector(mode = "list", length = length(tensors)),
    names = names(tensors)
  )

  pos <- 0L
  for (nm in names(tensors)) {
    meta <- tensor_meta(tensors[[nm]])
    meta$data_offsets <- c(pos, pos + size_from_meta(meta))
    pos <- meta$data_offsets[2]
    meta_[[nm]] <- meta
  }

  if (!is.null(metadata)) {
    meta_[["__metadata__"]] <- metadata
  }

  meta_
}

tensor_buffer <- function(x) {
  UseMethod("tensor_buffer")
}

tensor_buffer.torch_tensor <- function(x) {
  torch::buffer_from_torch_tensor(x)
}

tensor_meta <- function(x) {
  UseMethod("tensor_meta")
}

tensor_meta.torch_tensor <- function(x) {
  list(
    shape = x$shape,
    dtype = torch_dtype_to_safe(x$dtype)
  )
}

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
  } else {
    cli::cli_abort("Unsupported data type {.val {x}}")
  }
}

size_from_meta <- function(meta) {
  numel <- prod(meta$shape)

  el_size <- if (meta$dtype == "F32") {
    4L
  } else if (meta$dtype == "F16") {
    2L
  } else if (meta$dtype == "F64") {
    8L
  } else if (meta$dtype == "U8") {
    1L
  } else if (meta$dtype == "I8") {
    1L
  } else if (meta$dtype == "I16") {
    2L
  } else if (meta$dtype == "I32") {
    4L
  } else if (meta$dtype == "I64") {
    8L
  } else if (meta$dtype == "BOOL") {
    1L
  } else {
    cli::cli_abort("Unsupported dtype {.val {meta$dtype}}")
  }

  as.integer(numel*el_size)
}
