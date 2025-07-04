module Model

using SpecialFunctions

"""
    Parameters(in_subwords::Array{Float32, 2}, in_senses::Array{Float32, 3}, out::Array{Float32, 2}, as::Array{Float32, 2}, bs::Array{Float32, 2})

Keeps track of and maintains the model
parameters that we are optimizing.
"""
struct Parameters
  in_subwords::Array{Float32, 2}
  in_senses::Array{Float32, 3}
  out::Array{Float32, 2}
  ns::Array{Float32, 2}
end

function initialize(vector_dims::Int64, input_words::Int64, word_counts::Vector{Int64}, input_subwords::Int64, senses::Int64, tree_nodes::Int64, alpha::Float32)
  in_subwords = (rand(Float32, input_subwords, vector_dims) .- 0.5) ./ vector_dims
  in_senses = (rand(Float32, input_words, vector_dims, senses) .- 0.5) ./ vector_dims
  out = (rand(Float32, vector_dims, tree_nodes) .- 0.5) ./ vector_dims
  ns = zeros(Float32, input_words, senses)
  ns[:, 1] += word_counts
  model = Parameters(in_subwords, in_senses, out, ns)
  return model
end

function forward_pass(model::Parameters, input_word::Int64, input_subwords::Vector{Int32})::Tuple{Array{Float32, 2}, Array{Float32, 2}}
  latent_representation = model.in_senses[input_word, :, :]' .+ sum(model.in_subwords[input_subwords, :], dims=1)
  output = latent_representation * model.out
  return latent_representation, output
end

# function sense_likelihoods(model::Parameters)::Array{Float32, 2}
#   words, senses = size(model.ns) # todo deal with ns instead of as, bs
#   likelihoods = zeros(Float32, words, senses)
#   for sense in 1:senses
#     likelihoods[:, sense] = as[:, sense] ./ (as[:, sense] .+ bs[:, sense])
#   end
#   prod = ones(Float32, words)
#   for sense in 1:(senses-1)
#     prod *= bs[:, sense] ./ (as[:, sense] .+ bs[:, sense])
#     likelihoods[:, sense+1] *= prod
#   end
#   return likelihoods
# end

function compute_beta_parameters(ns::Vector{Float32}, α::Float32)
  as = ones(Float32, length(ns))
  bs = fill(α, length(ns))
  cumulative_ns = zeros(Float32, length(ns))
  for (i, n) in reverse(enumerate(ns))
    cumulative_ns[i] = cumulative_ns[min(i, end)] + n
    as[i] += n
    bs[i] += cumulative_ns[i]
  end
  return as, bs
end

function train(model::Parameters, training_data::Vector{Tuple{UInt64, Vector{UInt32}, Vector{UInt64}}}, batch_size::Int64, epochs::Int64, α::Float32)
  for epoch in 1:epochs
    minibatches = minibatches(training_data, batch_size)
    for (i, minibatch) in enumerate(minibatches)
      println("Epoch $epoch, batch $i/$(length(minibatches)).")
      sense_likelihoods = zeros(Float32, length(minibatch), senses)
      for (j, (word, subwords, context)) in enumerate(minibatch)
        ψs = zeros(Float32, senses)
        bψs = zeros(Float32, senses)
        latent, output = forward_pass(model, word, subwords)
        as, bs = compute_beta_parameters(model.ns[word, sense], α)
        for sense in 1:senses
          ψs[sense] = ψ(as[sense] + bs[sense])
          bψs[sense] = ψ(bs[sense]) - ψs[sense]
          logbeta_k = ψ(as[sense]) - ψs[sense]
          logbeta_complements = sum(bψs[r] for r in 1:(sense-1))
          sense_likelihoods[j, sense] = logbeta_k + logbeta_complements
        end
        for context_word in context
          sense_likelihoods[j, :] .+= compute_likelihood_result(context_word, output)
        end
      end
      sense_likelihoods .= exp.(sense_likelihoods)
      totals = sum(sense_likelihoods, dims=2)
      sense_likelihoods ./= totals
      for (j, (word, _, _)) in enumerate(minibatch)
        model.ns[word, :] = (1-λ) .* model.ns[word, :] .+ (λ*count[word]) .* sense_likelihoods[j, :]
      end
      # TODO update outs
      # TODO update theta_subwords
      # TODO update theta_words
    end
  end
end

end
