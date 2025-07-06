module AdaSubGram

using MKL

include("./hashing.jl")
include("./preprocessing.jl")
include("./dataset.jl")
include("./model.jl")
include("./hierarchical_softmax.jl")

const Filepath=String

function create_encodings(parameters::Filepath, output::Filepath)
  # TODO read file
  # TODO create dataset
  # TODO construct huffman tree
  # TODO training loop
  # TODO export embeddings
  # TODO export sense expectations
end

export create_encodings

end
