#' @title Reflection of supported frameworks
#' @description
#' A reflection of supported frameworks.
#' @export
safetensors_frameworks <- new.env()

safetensors_frameworks[["torch"]] <- list(
  constructor = torch_tensor_from_raw,
  packages = "torch"
)
