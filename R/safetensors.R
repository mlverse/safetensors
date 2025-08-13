#' Safe load a safetensors file
#'
#' Loads an safetensors file from disk.
#'
#' @param path Path to the file to load
#' @param framework Framework to load the data into. Currently supports "torch" and "pjrt"
#' @param ... Additional framework dependent arguments to pass to the tensor creation function.
#'
#' @returns A list with tensors in the file. The `metadata` attribute can be used
#' to find metadata the metadata header in the file.
#'
#' @examples
#' if (rlang::is_installed("torch") && torch::torch_is_installed()) {
#'   tensors <- list(x = torch::torch_randn(10, 10))
#'   temp <- tempfile()
#'   safe_save_file(tensors, temp)
#'   safe_load_file(temp, framework = "torch")
#' }
#'
#' @seealso [safetensors], [safe_save_file()]
#'
#' @export
safe_load_file <- function(path, ..., framework) {
  f <- safetensors$new(path, ..., framework = framework)
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
#' f <- safetensors$new(temp, framework = "torch")
#' f$get_tensor("x")
#' }
#'
#' @importFrom R6 R6Class
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
    #' @field args additional arguments for tensor creation
    args = NULL,
    #' @field max_offset the largest offset boundary that was visited. Mainly
    #' used in torch to find the end of the safetensors file.
    max_offset = 0L,
    #' @description
    #' Opens the connection with the file
    #' @param path Path to the file to load
    #' @param framework Framework to load the data into. Currently supports "torch" and "pjrt"
    #' @param ... (any)\cr
    #'   Additional, framework dependent, arguments to pass to use when creating the tensor.
    #'   For torch, this is the device, for pjrt the client.
    initialize = function(path, ..., framework) {
      self$framework <- validate_framework(framework)
      self$args <- list(...)

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

      self$metadata <- jsonlite::parse_json(
        rawToChar(raw_json),
        simplifyVector = TRUE
      )
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

      if (!self$framework %in% names(safetensors_frameworks)) {
        cli::cli_abort("Unsupported framework {.val {.self$framework}}")
      }

      rlang::exec(
        safetensors_frameworks[[self$framework]]$constructor,
        raw_tensor,
        meta,
        !!!self$args
      )
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

validate_framework <- function(x) {
  info <- safetensors_frameworks[[x]]

  if (is.null(info)) {
    cli::cli_abort("Unsupported framework {.val {x}}")
  }
  rlang::check_installed(info$packages, reason = "for loading {x} tensors.")
  x
}
