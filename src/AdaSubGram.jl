module AdaSubGram

using MKL

include("./hashing.jl")
include("./preprocessing.jl")
include("./dataset.jl")
include("./model.jl")
include("./hierarchical_softmax.jl")
include("./export.jl")

const Filepath=String

function create_encodings(parameters::Filepath, output::Filepath)
  documents = readlines(parameters)[1:500]
  dims = 100
  α = 0.1f0
  λ = 0.05f0
  senses = 10
  context = 8
  subword_truncation = 1_000_000
  s_min = 4
  s_max = 7
  batch_size = 128
  epochs = 5
  normalized_documents = AdaSubGram.Preprocessing.normalize.(documents)
  tokenized_documents = AdaSubGram.Preprocessing.tokenize.(normalized_documents)
  dataset, counts, labels = AdaSubGram.Dataset.create_dataset(tokenized_documents, context, s_min, s_max, UInt32(subword_truncation))
  nodes_decisions = AdaSubGram.HuffmanTree.huffman_paths(counts)
  model = AdaSubGram.Model.initialize(dims, counts, subword_truncation, senses)
  AdaSubGram.Model.train(model, dataset, nodes_decisions, batch_size, epochs, α, λ)
  @views AdaSubGram.Export.embeddings(output, labels, model.in_subwords, model.in_senses, s_min, s_max, UInt32(subword_truncation))
end

export create_encodings

end
