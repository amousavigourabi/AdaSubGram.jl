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

struct Settings
  dims::Int64
  senses::Int64
  context::Int64
  α::Float32
  epochs::Int64
  s_min::Int64
  s_max::Int64
  subword_truncation::Int64
  η_1::Float32
  η_2::Float32
end

function settings(dims::Int64=300, senses::Int64=10, context::Int64=8, α::Float32=0.15f0, epochs::Int64=3, s_min::Int64=4, s_max::Int64=7, subword_truncation::Int64=1_000_000, η_1::Float32=0.025f0, η_2::Float32=NaN32)
  return Settings(dims, senses, context, α, epochs, s_min, s_max, subword_truncation, η_1, η_2)
end

function create_encodings(input::Filepath, output::Filepath)
  create_encodings(input, output, settings())
end

function create_encodings(input::Filepath, output::Filepath, settings::Settings)
  documents = readlines(input)
  tokenized_documents = Vector{Vector{String}}(undef, size(documents))
  @threads for i in eachindex(documents)
    @inbounds tokenized_documents[i] = AdaSubGram.Preprocessing.tokenize(AdaSubGram.Preprocessing.normalize(documents[i]))
  end
  dataset, counts, labels = AdaSubGram.Dataset.create_dataset(tokenized_documents, settings.context, settings.s_min, settings.s_max, UInt32(settings.subword_truncation))
  nodes_decisions = AdaSubGram.HuffmanTree.huffman_paths(counts)
  model = AdaSubGram.Model.initialize(settings.dims, counts, settings.subword_truncation, settings.senses)
  @inbounds max_nodes = maximum(length, nodes_decisions[1])
  train_settings = AdaSubGram.Model.settings(settings.α, settings.epochs, settings.η_1, settings.η_2)
  final_loss = AdaSubGram.Model.train(model, dataset, nodes_decisions, train_settings, max_nodes)
  @views AdaSubGram.Export.embeddings(output, labels, model.in_subwords, model.in_senses, settings.s_min, settings.s_max, UInt32(settings.subword_truncation))
  return final_loss
end

export create_encodings

end
