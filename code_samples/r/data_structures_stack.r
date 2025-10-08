# Data Structures: Stack
# AI/ML Training Sample

Stack <- setRefClass(
    "Stack",
    fields = list(data = "character"),
    methods = list(
        initialize = function() {
            data <<- ""
        },
        process = function(input) {
            data <<- input
        },
        getData = function() {
            return(data)
        },
        validate = function() {
            return(nchar(data) > 0)
        }
    )
)

# Example usage
instance <- Stack$new()
instance$process("example")
cat("Data:", instance$getData(), "\n")
cat("Valid:", instance$validate(), "\n")
