#' Safe load a safetensors file
#'
#' Loads an safetensors file from disk.
#'
#' @param path Path to the file to load
#' @param ... Unused
#' @param framework Framework to load the data into. Currently only torch is supported
#' @param device Device to copy data once loaded
#'
#' @returns A list with tensors in the file. The `metadata` attribute can be used
#' to find metadata the metadata header in the file.
#'
#' @examples
#' if (rlang::is_installed("torch") && torch::torch_is_installed()) {
#'   tensors <- list(x = torch::torch_randn(10, 10))
#'   temp <- tempfile()
#'   safe_save_file(tensors, temp)
#'   safe_load_file(temp)
#' }
#'
#' @seealso [safetensors], [safe_save_file()]
#'
#' @export
safe_load_file <- function(path, ..., framework = "torch", device = "cpu") {
  f <- safetensors$new(path, framework = framework, device = device)
  nms <- f$keys()
  output <- structure(
    vector(length = length(nms), mode = "list"),
    names = nms,
    metadata = f$metadata
  )
  for (key in nms) {
    output[[key]] <- f$get_tensor(key)
  }
  attr(output, "max_offset") <- f$max_offset
  output
}

#' Low level control over safetensors files
#'
#' Allows opening a connection to a safetensors file and query the tensor names,
#' metadata, etc.
#' Opening a connection only reads the file metadata into memory.
#' This allows for more fined grained control over reading.
#'
#' @examples
#' if (rlang::is_installed("torch") && torch::torch_is_installed()) {
#' tensors <- list(x = torch::torch_randn(10, 10))
#' temp <- tempfile()
#' safe_save_file(tensors, temp)
#' f <- safetensors$new(temp)
#' f$get_tensor("x")
#' }
#'
#' @export
safetensors <- R6::R6Class(
  "safetensors",
  public = list(
    #' @field con the connection object with the file
    con = NULL,
    #' @field metadata an R list containing the metadata header in the file
    metadata = NULL,
    #' @field framework the framework used to return the tensors
    framework = NULL,
    #' @field device the device to where tensors are copied
    device = NULL,
    #' @field max_offset the largest offset boundary that was visited. Mainly
    #' used in torch to find the end of the safetensors file.
    max_offset = 0L,
    #' @description
    #' Opens the connection with the file
    #' @param path Path to the file to load
    #' @param ... Unused
    #' @param framework Framework to load the data into. Currently only torch is supported
    #' @param device Device to copy data once loaded
    initialize = function(path, ..., framework = "torch", device = "cpu") {
      self$framework <- validate_framework(framework)
      self$device <- device

      # read in the metadata and store it
      if (is.raw(path)) {
        self$con <- rawConnection(path, open = "rb")
      } else if (is.character(path)) {
        self$con <- file(path, "rb")
      } else if (inherits(path, "connection")) {
        # safetensors has no responsability over this connection as this was
        # created efore passing to it.
        private$close_con <- FALSE
        self$con <- path
      }

      metadata_size <- readBin(self$con, what = integer(), n = 1, size = 8)
      raw_json <- readBin(self$con, what = "raw", n = metadata_size)

      self$metadata <- jsonlite::fromJSON(rawToChar(raw_json))
      private$byte_buffer_begin <- 8L + metadata_size
    },
    #' @description
    #' Get the keys (tensor names) in the file
    keys = function() {
      keys <- names(self$metadata)
      keys[keys != "__metadata__"]
    },
    #' @description
    #' Get a tensor from its name
    #' @param name Name of the tensor to load
    get_tensor = function(name) {
      meta <- self$metadata[[name]]

      offset_start <- private$byte_buffer_begin + meta$data_offsets[1]
      offset_length <- meta$data_offsets[2] - meta$data_offsets[1]
      self$max_offset <- max(self$max_offset, offset_start + offset_length)

      seek(self$con, offset_start)
      raw_tensor <- readBin(self$con, what = "raw", n = offset_length)

      if (self$framework == "torch") {
        torch_tensor_from_raw(raw_tensor, meta, self$device)
      } else {
        cli::cli_abort("Unsupported framework {.val {.self$framework}}")
      }
    }
  ),
  private = list(
    byte_buffer_begin = 0L,
    close_con = TRUE,
    finalize = function() {
      if (private$close_con) {
        close(self$con)
      }
    }
  )
)

torch_tensor_from_raw <- function(raw, meta, device) {
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
  switch (
    x,
    "F16" = "float16",
    "F32" = "float",
    "F64" = "float64",
    "BOOL" = "bool",
    "U8" = "uint8",
    "I8" = "int8",
    "I16" = "int16",
    "I32" = "int32",
    "I64" = "int64"
  )
}

validate_framework <- function(x) {
  if (!x %in% c("torch")) {
    cli::cli_abort("Unsupported framework {.val {x}}")
  }
  if (x == "torch") {
    rlang::check_installed(x, reason = "for loading torch tensors.")
  }
  x
}
