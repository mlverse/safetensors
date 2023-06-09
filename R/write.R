#' Writes a list of tensors to the safetensors format
#'
#' @param tensors A named list of tensors. Currently only torch tensors are supported.
#' @param path The path to save the tensors to. It can also be a binary connection, as eg,
#'   created with `file()`.
#' @param ... Currently unused.
#' @param metadata An optional string that is added to the file header. Possibly
#'   adding additional description to the weights.
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
  }
}

size_from_meta <- function(meta) {
  numel <- prod(meta$shape)

  el_size <- if (meta$dtype == "F32") {
    4L
  } else if (meta$dtype == "F16") {
    2L
  }

  as.integer(numel*el_size)
}
