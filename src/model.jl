module Model

"""
    Parameters(in_subwords::Array{Float32, 2}, in_senses::Array{Float32, 3}, out::Array{Float32, 2}, as::Array{Float32, 2}, bs::Array{Float32, 2})

Keeps track of and maintains the model
parameters that we are optimizing.
"""
struct Parameters
  in_subwords::Array{Float32, 2}
  in_senses::Array{Float32, 3}
  out::Array{Float32, 2}
  as::Array{Float32, 2}
  bs::Array{Float32, 2}
end

function initialize_model(vector_dims::Int64, input_words::Int64, word_counts::Vector{Int64}, input_subwords::Int64, senses::Int64, tree_nodes::Int64, alpha::Float32)
  in_subwords = rand(Float32, input_subwords, vector_dims) .- 0.5 ./ vector_dims
  in_senses = rand(Float32, input_words, senses, vector_dims) .- 0.5 ./ vector_dims
  out = rand(Float32, vector_dims, tree_nodes) .- 0.5 ./ vector_dims
  as = Array{Float32}(1.0, input_words, senses)
  as[1, :] += word_counts
  bs = Array{Float32}(alpha, input_words, senses)
  model = Parameters(in_subwords, in_senses, out, as, bs)
  return model
end

export initialize_model

end
