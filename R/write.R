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
#'   safe_load_file(temp, framework = "torch")
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
    on.exit(
      {
        close(con)
      },
      add = TRUE
    )
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
  on.exit(
    {
      close(con)
    },
    add = TRUE
  )
  safe_save_file(tensors, con, metadata = metadata)
  rawConnectionValue(con)
}

write_safe <- function(tensors, metadata, con) {
  meta <- make_meta(tensors, metadata)
  meta_json <- meta_to_json(meta)
  meta_raw <- charToRaw(meta_json)
  # write the metadatasize as a 64bit int

  writeBin(length(meta_raw), con = con, size = 8L)
  writeBin(meta_raw, con = con)
  # Use index-based iteration to handle empty string keys
  for (i in seq_along(tensors)) {
    buf <- safe_tensor_buffer(tensors[[i]])
    writeBin(buf, con = con)
  }
}

# Custom JSON serializer that preserves empty string keys
# jsonlite::toJSON converts "" to "1", which breaks safetensors compatibility
meta_to_json <- function(meta) {
  nms <- names(meta)
  # Check if we need custom handling (empty keys present)
  if (!any(nchar(nms) == 0)) {
    return(jsonlite::toJSON(meta, auto_unbox = TRUE))
  }

  # Manual JSON construction to preserve empty string keys
  parts <- character(length(meta))
  for (i in seq_along(meta)) {
    nm <- nms[i]
    val <- jsonlite::toJSON(meta[[i]], auto_unbox = TRUE)
    parts[i] <- sprintf("\"%s\":%s", nm, val)
  }
  paste0("{", paste(parts, collapse = ","), "}")
}

make_meta <- function(tensors, metadata) {
  nms <- names(tensors)
  meta_ <- structure(
    vector(mode = "list", length = length(tensors)),
    names = nms
  )

  if (!is.null(metadata)) {
    meta_[["__metadata__"]] <- validate_metadata(metadata)
  }

  # Offsets accumulate past .Machine$integer.max for multi-GB files;
  # keep them as doubles (exactly representable well past 2^31)
  pos <- 0
  # Use index-based access for tensors to handle empty string keys
  # R's list[[""]] returns NULL even when "" is a valid key
  for (i in seq_along(tensors)) {
    meta <- safe_tensor_meta(tensors[[i]])
    meta$data_offsets <- c(pos, pos + size_from_meta(meta))
    pos <- meta$data_offsets[2]
    # Assign to the pre-named slot by index, preserving the original name
    meta_[[i]] <- meta
  }

  # Restore original names (index assignment can corrupt empty string names)
  names(meta_) <- nms

  meta_
}

#' @title Get raw buffer from a tensor
#' @description
#' Convert a tensor object to a raw buffer in the formated expected by safetensors.
#' @param x (any)\cr
#'   Tensor object.
#' @returns (`raw`)
#' @export
safe_tensor_buffer <- function(x) {
  UseMethod("safe_tensor_buffer")
}

#' @title Get metadata from a tensor
#' @description
#' Get the metadata from a tensor.
#' @param x (any)\cr
#'   Tensor object.
#' @returns (`list`)
#' @export
safe_tensor_meta <- function(x) {
  UseMethod("safe_tensor_meta")
}

size_from_meta <- function(meta) {
  numel <- prod(as.numeric(meta$shape))

  el_size <- if (meta$dtype == "F32") {
    4L
  } else if (meta$dtype == "F16") {
    2L
  } else if (meta$dtype == "BF16") {
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
  } else if (meta$dtype == "U16") {
    2L
  } else if (meta$dtype == "U32") {
    4L
  } else if (meta$dtype == "U64") {
    8L
  } else {
    cli::cli_abort("Unsupported dtype {.val {meta$dtype}}")
  }

  # Doubles, not integers: tensor byte sizes and cumulative offsets can
  # exceed .Machine$integer.max
  numel * el_size
}

validate_metadata <- function(x) {
  if (!rlang::is_list(x)) {
    cli::cli_abort("{.arg metadata} must be a list.")
  }
  if (!rlang::is_named(x)) {
    cli::cli_abort("{.arg metadata} must be a named list.")
  }
  lapply(x, function(item) {
    if (!rlang::is_scalar_character(item)) {
      cli::cli_abort(
        "{.arg metadata} must be a named list of scalar characters."
      )
    }
  })
  x
}
