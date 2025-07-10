module AdaSubGram

using MKL
using Base.Threads

include("./hashing.jl")
include("./preprocessing.jl")
include("./dataset.jl")
include("./model.jl")
include("./hierarchical_softmax.jl")
include("./export.jl")

const Filepath=String

function create_encodings(parameters::Filepath, output::Filepath)
  documents = readlines(parameters)[1:1000]
  dims = 100
  α = 0.1f0
  senses = 10
  context = 8
  subword_truncation = 1_000_000
  s_min = 4
  s_max = 7
  epochs = 5
  tokenized_documents = Vector{Vector{String}}(undef, size(documents))
  @threads for i in eachindex(documents)
    tokenized_documents[i] = AdaSubGram.Preprocessing.tokenize(AdaSubGram.Preprocessing.normalize(documents[i]))
  end
  dataset, counts, labels = AdaSubGram.Dataset.create_dataset(tokenized_documents, context, s_min, s_max, UInt32(subword_truncation))
  nodes_decisions = AdaSubGram.HuffmanTree.huffman_paths(counts)
  model = AdaSubGram.Model.initialize(dims, counts, subword_truncation, senses)
  AdaSubGram.Model.train(model, dataset, nodes_decisions, epochs, α)
  @views AdaSubGram.Export.embeddings(output, labels, model.in_subwords, model.in_senses, s_min, s_max, UInt32(subword_truncation))
end

export create_encodings

end
