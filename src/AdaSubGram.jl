module AdaSubGram

using MKL

include("./hashing.jl")
include("./preprocessing.jl")
include("./dataset.jl")
include("./model.jl")
include("./hierarchical_softmax.jl")

const Filepath=String

function create_encodings(parameters::Filepath, output::Filepath)
  documents = readlines(parameters)
  dims = 100
  α = 0.5f0
  λ = 0.02f0
  senses = 6
  context = 10
  subword_truncation = 1_000_000
  s_min = 4
  s_max = 7
  batch_size = 32
  epochs = 10
  normalized_documents = AdaSubGram.Preprocessing.normalize.(documents)
  tokenized_documents = AdaSubGram.Preprocessing.tokenize.(normalized_documents)
  dataset, counts = AdaSubGram.Dataset.create_dataset(tokenized_documents, context, s_min, s_max, UInt32(subword_truncation), UInt64(100_000))
  nodes, decisions = AdaSubGram.HuffmanTree.huffman_paths(counts)
  model = AdaSubGram.Model.initialize(dims, counts, subword_truncation, senses)
  AdaSubGram.Model.train(model, dataset, collect(zip(nodes, decisions)), batch_size, epochs, α, λ)
  # TODO export embeddings
  # TODO export sense expectations
end

export create_encodings

end
