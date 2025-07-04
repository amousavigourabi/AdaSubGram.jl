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
  as = ones(Float32, input_words, senses)
  as[1, :] += word_counts
  bs = Array{Float32}(alpha, input_words, senses)
  model = Parameters(in_subwords, in_senses, out, as, bs)
  return model
end

function prototypes(model::Parameters, input_word::Int64, input_subwords::Vector{Int64})::Tuple{Array{Float32, 2}, Array{Float32, 2}}
  latent_representation = model.in_senses[input_word, :, :] + sum(model.in_subwords[input_subwords, :], dims=2)
  output = latent_representation * model.out
  return latent_representation, output
end

function sense_likelihoods(model::Parameters)::Array{Float32, 2}
  words, senses = size(model.as)
  likelihoods = zeros(Float32, words, senses)
  for sense in 1:senses
    likelihoods[:, sense] = as[:, sense] / (as[:, sense] + bs[:, sense])
  end
  prod = ones(Float32, words)
  for sense in 1:(senses-1)
    prod *= bs[:, sense] / (as[:, sense] + bs[:, sense])
    likelihoods[:, sense+1] *= prod
  end
  return likelihoods
end

export initialize_model

end
